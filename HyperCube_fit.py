"""Qt-free per-spaxel fit kernel for HyperCube.

This module holds the pure compute used by both the serial and the parallel
("Fit Cube") paths. It imports ONLY numpy / pandas / lmfit / astropy and the
already-Qt-free HyperCube_ModelFunctions and HyperCube_pPXF modules, so it can be
imported cleanly inside multiprocessing workers (which, under the macOS 'spawn'
start method, re-import the target module) WITHOUT pulling in PyQt5 or launching
the GUI.

The functions here take everything they need as arguments (no module globals from
HyperCube.py, no `self`, no Qt), so a worker process and the main process produce
byte-identical results for the same inputs.
"""

import ast
import os
from multiprocessing import shared_memory

import numpy as np
import pandas as pd
from lmfit import Model, Parameters

import HyperCube_ModelFunctions

try:
    import HyperCube_pPXF as hcppxf
except Exception:        # pPXF optional
    hcppxf = None


C_KMS = 299792.458


# ── small helpers (mirrors of the ones in HyperCube.py) ──────────────────────
def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _as_float_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            return []
    try:
        return [float(z) for z in v]
    except Exception:
        return []


# ── model construction ───────────────────────────────────────────────────────
def build_model(n_regions, n_lines):
    """Rebuild the piecewise (continuum + Gaussians) lmfit Model. Matches the
    construction in HyperCube.fit_cube."""
    maker = HyperCube_ModelFunctions.PiecewiseModel(n_regions=n_regions,
                                                    n_gaussians=n_lines)
    return Model(maker.model_function)


# ── component pairs / staged fit (sequential core→outflow) ───────────────────
def component_pairs(df):
    """Model-order (1-based) index pairs of two-component lines sharing a rest
    wavelength, ordered narrow-first by σ guess."""
    if 'Rest Wavelength' not in df.columns:
        return []
    groups = {}
    for idx, (_, r) in enumerate(df.iterrows(), start=1):
        rw = _safe_float(r.get('Rest Wavelength'))
        sig = _safe_float(r.get('Sigma_0'))
        if np.isfinite(rw):
            groups.setdefault(round(rw, 4), []).append((idx, sig))
    pairs = []
    for members in groups.values():
        if len(members) == 2:
            members.sort(key=lambda t: (t[1] if np.isfinite(t[1]) else np.inf))
            pairs.append((members[0][0], members[1][0]))
    return pairs


def staged_fit(model, y, params, wavelengths, max_nfev, df):
    """Sequential core→outflow fit (breaks the narrow/broad degeneracy). Falls
    back to a single joint fit if there are no narrow/broad pairs or on error."""
    try:
        pairs = component_pairs(df)
    except Exception:
        pairs = []
    if not pairs:
        return model.fit(y, params, x=wavelengths, max_nfev=max_nfev)

    try:
        broad_amps = {f'amp{i_b}' for (_i_n, i_b) in pairs}
        cont_prefixes = ('slope', 'intercept', 'polyc', 'knoty')
        orig_vary = {name: p.vary for name, p in params.items()}

        # Stage 1 — core only: suppress free broad amplitudes.
        p1 = params.copy()
        for k in broad_amps:
            if k in p1 and not p1[k].expr:
                p1[k].set(value=0.0, vary=False)
        r1 = model.fit(y, p1, x=wavelengths, max_nfev=max_nfev)

        # Stage 2 — broad on the residual: freeze core + continuum, free broad.
        p2 = r1.params.copy()
        for name, p in p2.items():
            if name in broad_amps:
                if not p.expr:
                    p.set(value=params[name].value, vary=True)
                    if params[name].min is not None:
                        p.min = params[name].min
                    if params[name].max is not None:
                        p.max = params[name].max
            elif name.startswith('offset_vel') or name.startswith('ratio'):
                continue
            elif (name.startswith(('amp', 'cen', 'sigma')) or
                  name.startswith(cont_prefixes)):
                if p.expr is None:
                    p.vary = False
        r2 = model.fit(y, p2, x=wavelengths, max_nfev=max_nfev)

        # Stage 3 — joint polish from the staged solution.
        p3 = r2.params.copy()
        for name, p in p3.items():
            if p.expr is None and name in orig_vary:
                p.vary = orig_vary[name]
        return model.fit(y, p3, x=wavelengths, max_nfev=max(64, max_nfev // 2))
    except Exception as e:
        print(f"Staged fit fell back to joint fit: {type(e).__name__}: {e}")
        return model.fit(y, params, x=wavelengths, max_nfev=max_nfev)


# ── calibrated goodness-of-fit metrics ───────────────────────────────────────
def fit_quality_metrics(spectrum, wavelengths, result, flux_scale, df, df_cont):
    """Calibrated, scale-free goodness-of-fit statistics for one spaxel
    (see HyperCube._fit_quality_metrics for the full rationale)."""
    nan = float('nan')
    out = {'qa_noise': nan, 'qa_chisq_cont': nan, 'qa_chisq_core': nan,
           'qa_core_cont_ratio': nan, 'qa_signed_resid_z': nan, 'qa_runs_z': nan}
    try:
        model = np.asarray(result.best_fit, dtype=float) * flux_scale
        lam = np.asarray(wavelengths, dtype=float)
        data = np.asarray(spectrum, dtype=float)
        if model.shape != data.shape or model.shape != lam.shape:
            return out
        resid = data - model
        finite = np.isfinite(resid) & np.isfinite(lam)

        in_fit = np.zeros_like(lam, dtype=bool)
        nreg = len(df_cont)
        for r in range(1, nreg + 1):
            ks, ke = f'x{r}_start', f'x{r}_end'
            if ks in result.params and ke in result.params:
                x0 = float(result.params[ks].value)
                x1 = float(result.params[ke].value)
                in_fit |= (lam >= min(x0, x1)) & (lam <= max(x0, x1))
        if not in_fit.any():
            in_fit = finite.copy()
        in_fit &= finite

        core = np.zeros_like(lam, dtype=bool)
        nlines = len(np.unique(df['Line_ID'])) if 'Line_ID' in df.columns else 0
        for i in range(1, nlines + 1):
            ck, sk = f'cen{i}', f'sigma{i}'
            if ck in result.params and sk in result.params:
                cen = float(result.params[ck].value)
                sig = abs(float(result.params[sk].value))
                if np.isfinite(cen) and np.isfinite(sig) and sig > 0:
                    core |= np.abs(lam - cen) <= 2.5 * sig
                    p_cen = result.params[ck]
                    cen_lo = (float(p_cen.min)
                              if p_cen.min is not None and np.isfinite(float(p_cen.min))
                              else cen)
                    cen_hi = (float(p_cen.max)
                              if p_cen.max is not None and np.isfinite(float(p_cen.max))
                              else cen)
                    p_sig = result.params[sk]
                    sig_lo = (float(p_sig.min)
                              if p_sig.min is not None and np.isfinite(float(p_sig.min))
                              else sig)
                    sig_pad = max(sig, sig_lo) * 2.5
                    core |= (lam >= cen_lo - sig_pad) & (lam <= cen_hi + sig_pad)
        core &= in_fit
        cont = in_fit & ~core

        r_core, r_cont = resid[core], resid[cont]
        n_core, n_cont = r_core.size, r_cont.size
        if n_cont >= 5:
            noise = 1.4826 * np.median(np.abs(r_cont - np.median(r_cont)))
            out['qa_noise'] = float(noise)
            if noise > 0:
                out['qa_chisq_cont'] = float(np.mean(r_cont**2) / noise**2)
                if n_core >= 1:
                    out['qa_chisq_core'] = float(np.mean(r_core**2) / noise**2)
                    out['qa_signed_resid_z'] = float(
                        np.sum(r_core) / (noise * np.sqrt(n_core)))
            mc = np.mean(r_cont**2)
            if n_core >= 1 and mc > 0:
                out['qa_core_cont_ratio'] = float(np.mean(r_core**2) / mc)

        if n_core >= 10:
            signs = np.sign(r_core)
            signs = signs[signs != 0]
            npos = int((signs > 0).sum()); nneg = int((signs < 0).sum())
            ntot = npos + nneg
            if npos > 0 and nneg > 0 and ntot > 1:
                runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
                mu = 2.0 * npos * nneg / ntot + 1.0
                var = (2.0 * npos * nneg * (2.0 * npos * nneg - ntot)
                       / (ntot**2 * (ntot - 1)))
                if var > 0:
                    out['qa_runs_z'] = float((runs - mu) / np.sqrt(var))
    except Exception as e:
        print(f"QA-metrics warning: {type(e).__name__}: {e}")
    return out


# ── the per-spaxel full-model fit ────────────────────────────────────────────
def fit_one_spaxel(spectrum, stellar_baseline, wavelengths, params_to_use, model,
                   df, df_cont, z, max_nfev, sequential, spaxel_xy, ra, dec,
                   stellar_kin):
    """Fit the full spectral model (continuum of any type per region + a Gaussian
    per line) to ONE spaxel and return a list of per-line result-row dicts (or a
    single error row). Mirrors the body of HyperCube.fit_spaxel exactly, but with
    all state passed in.

    spectrum          : raw 1D flux for the spaxel (NaNs allowed).
    stellar_baseline  : array subtracted from spectrum (zeros if no stellar region).
    stellar_kin       : {region_id: {'V','sigma','h3','h4','scale'}} for stellar regions.
    spaxel_xy         : (x, y); ra/dec : sky coords for the result rows.
    """
    cx, cy = int(spaxel_xy[0]), int(spaxel_xy[1])
    rows = []

    spectrum = np.nan_to_num(np.asarray(spectrum, dtype=float))
    spectrum = spectrum - np.asarray(stellar_baseline, dtype=float)

    # Flux rescaling (keeps the Jacobian non-singular for ~1e-18 fluxes).
    flux_scale = (np.nanpercentile(np.abs(spectrum[spectrum != 0]), 95)
                  if np.any(spectrum != 0) else 1.0)
    if flux_scale == 0 or not np.isfinite(flux_scale):
        flux_scale = 1.0
    spectrum_scaled = spectrum / flux_scale

    params_scaled = params_to_use.copy()
    for pname, param in params_scaled.items():
        if param.expr:
            continue
        if pname.startswith('amp'):
            param.set(value=param.value / flux_scale)
            if param.min is not None:
                param.min = param.min / flux_scale
            if param.max is not None:
                param.max = param.max / flux_scale
        elif pname.startswith('intercept'):
            param.set(value=param.value / flux_scale)
        elif pname.startswith('knoty'):
            param.set(value=param.value / flux_scale)
            if param.min is not None and np.isfinite(param.min):
                param.min = param.min / flux_scale
            if param.max is not None and np.isfinite(param.max):
                param.max = param.max / flux_scale
        elif pname.startswith('polyc'):
            param.set(value=param.value / flux_scale)
            if param.min is not None and np.isfinite(param.min):
                param.min = param.min / flux_scale
            if param.max is not None and np.isfinite(param.max):
                param.max = param.max / flux_scale

    c = C_KMS

    try:
        if sequential:
            result = staged_fit(model, spectrum_scaled, params_scaled,
                                wavelengths, max_nfev, df)
        else:
            result = model.fit(spectrum_scaled, params_scaled, x=wavelengths,
                               max_nfev=max_nfev)

        qa = fit_quality_metrics(spectrum, wavelengths, result, flux_scale, df, df_cont)

        cont_map = {f"x{i + 1}_start": i + 1 for i in range(len(df_cont))}

        for line_idx, line in enumerate(df.itertuples(), start=1):
            line_id = line.Line_ID
            line_name = line.Line_Name
            rest_wavelength = float(df.loc[df['Line_ID'] == line_id, 'Rest Wavelength'].iloc[0])

            region_id = line.region_ID
            region_index = df_cont[df_cont['region_ID'] == region_id].index[0] + 1

            cont_params = {
                f'cont_region{region_index}_x_start': params_to_use[f'x{region_index}_start'].value,
                f'cont_region{region_index}_x_end': params_to_use[f'x{region_index}_end'].value,
                f'cont_region{region_index}_slope_init': params_to_use[f'slope{region_index}'].init_value,
                f'cont_region{region_index}_slope_fit': result.params[f'slope{region_index}'].value * flux_scale,
                f'cont_region{region_index}_intercept_init': params_to_use[f'intercept{region_index}'].init_value,
                f'cont_region{region_index}_intercept_fit': result.params[f'intercept{region_index}'].value * flux_scale
            }

            _nk = int(params_to_use[f'NK{region_index}'].value) if f'NK{region_index}' in params_to_use else 0
            _np_ = int(params_to_use[f'NP{region_index}'].value) if f'NP{region_index}' in params_to_use else 0
            _mrow = df_cont[df_cont['region_ID'] == region_id]
            _model_ctype = (str(_mrow.iloc[0]['cont_type'])
                            if len(_mrow) and 'cont_type' in df_cont.columns else 'linear')
            if _model_ctype == 'stellar':
                _ctype_out = 'stellar'
                _scache = stellar_kin.get(region_id) or stellar_kin.get(int(region_id)) or {}
                cont_params[f'cont_region{region_index}_stellar_V'] = _safe_float(_scache.get('V'))
                cont_params[f'cont_region{region_index}_stellar_sigma'] = _safe_float(_scache.get('sigma'))
                cont_params[f'cont_region{region_index}_stellar_h3'] = _safe_float(_scache.get('h3'))
                cont_params[f'cont_region{region_index}_stellar_h4'] = _safe_float(_scache.get('h4'))
                cont_params[f'cont_region{region_index}_stellar_scale'] = _safe_float(_scache.get('scale'))
            else:
                _ctype_out = 'poly' if _np_ >= 1 else ('spline' if _nk >= 2 else 'linear')
            cont_params[f'cont_region{region_index}_cont_type'] = _ctype_out
            if _np_ >= 1:
                cont_params[f'cont_region{region_index}_poly_degree'] = _np_ - 1
                cont_params[f'cont_region{region_index}_poly_coef_init'] = [
                    params_to_use[f'polyc{region_index}_{j}'].init_value for j in range(_np_)]
                cont_params[f'cont_region{region_index}_poly_coef_fit'] = [
                    result.params[f'polyc{region_index}_{j}'].value * flux_scale for j in range(_np_)]
            if _nk >= 2:
                cont_params[f'cont_region{region_index}_knots_x'] = [
                    params_to_use[f'knotx{region_index}_{k}'].value for k in range(_nk)]
                cont_params[f'cont_region{region_index}_knots_y_init'] = [
                    params_to_use[f'knoty{region_index}_{k}'].init_value for k in range(_nk)]
                cont_params[f'cont_region{region_index}_knots_y_fit'] = [
                    result.params[f'knoty{region_index}_{k}'].value * flux_scale for k in range(_nk)]

            if region_index < len(df_cont):
                cont_params.update({
                    f'cont_region{region_index}_x_int_start': params_to_use[f'x_int_{region_index}_start'].value,
                    f'cont_region{region_index}_x_int_end': params_to_use[f'x_int_{region_index}_end'].value,
                    f'cont_region{region_index}_slope_int_init': params_to_use[f'slope_int_{region_index}'].init_value,
                    f'cont_region{region_index}_slope_int_fit': result.params[f'slope_int_{region_index}'].value,
                })

            amp_key = f'amp{line_idx}'
            cen_key = f'cen{line_idx}'
            sigma_key = f'sigma{line_idx}'

            if np.isfinite(rest_wavelength):
                vel_init = c * ((params_to_use[cen_key].init_value / (rest_wavelength * (z + 1))) - 1)
                vel_fit = c * ((result.params[cen_key].value / (rest_wavelength * (z + 1))) - 1)
            else:
                vel_init = vel_fit = np.nan

            fit_entry = {
                'spaxel_x': cx,
                'spaxel_y': cy,
                'RA': float(ra),
                'Dec': float(dec),
                'region_ID': region_id,
                'LineName': line_name,
                'LineID': line_id,

                'amp_init': params_to_use[amp_key].init_value,
                'amp_fit': result.params[amp_key].value * flux_scale,
                'amp_std': (result.params[amp_key].stderr or 0) * flux_scale,

                'cen_init': params_to_use[cen_key].init_value,
                'cen_fit': result.params[cen_key].value,
                'cen_std': result.params[cen_key].stderr,

                'vel_init': vel_init,
                'vel_fit': vel_fit,
                'rest_wavelength': rest_wavelength,

                'sigma_init': params_to_use[sigma_key].init_value,
                'sigma_fit': result.params[sigma_key].value,
                'sigma_std': result.params[sigma_key].stderr,

                'BIC': result.bic,
                'rchisq': result.redchi,
                'success': result.success
            }

            fit_entry.update(cont_params)
            fit_entry.update(qa)
            rows.append(fit_entry)

    except Exception as e:
        rows.append({
            'spaxel_x': cx,
            'spaxel_y': cy,
            'fit_success': False,
            'error': str(e)
        })
        print(f"Fit error for spaxel ({cx},{cy}): {type(e).__name__}: {e}")

    return rows


# ── stellar (pPXF) per-spaxel fit: kinematics + baseline only ────────────────
def fit_stellar_one(spectrum, wavelengths, z, R, prep):
    """Run pPXF for one spaxel/region. Returns (kin, baseline):
      kin      : {'region_ID','V','sigma','h3','h4','scale','chi2'} or None on failure
      baseline : the stellar continuum over `wavelengths` to subtract (zeros on failure)
    The optimal-template cache is intentionally NOT returned (kinematics-only mode);
    overlays recompute it lazily in the main process when a spaxel is viewed."""
    lam = np.asarray(wavelengths, float)
    if hcppxf is None:
        return None, np.zeros_like(lam)
    flux = np.nan_to_num(np.asarray(spectrum, dtype=float))
    try:
        res = hcppxf.fit_stellar(
            flux, wavelengths, z, R, prep['lib'],
            fit_range=prep['fit_range'], mask_centroids=prep['mask'],
            moments=prep['moments'], degree=-1, mdegree=10,
            sigma_guess=150.0, velscale=prep['velscale'])
    except Exception:
        return None, np.zeros_like(lam)

    cache = res['cache']
    x1, x2 = float(prep['fit_range'][0]), float(prep['fit_range'][1])
    try:
        y = hcppxf.eval_stellar_baseline(
            cache, res['V'], res['sigma'], res['h3'], res['h4'], res['scale'], lam)
        baseline = np.nan_to_num(np.asarray(y, dtype=float))
        mask = (lam >= x1) & (lam <= x2)
        baseline = np.where(mask, baseline, 0.0)
    except Exception:
        baseline = np.zeros_like(lam)

    kin = {'region_ID': int(prep['rid']), 'V': res['V'], 'sigma': res['sigma'],
           'h3': res['h3'], 'h4': res['h4'], 'scale': res['scale'],
           'chi2': res['chi2']}
    return kin, baseline


# ── multiprocessing worker (spawn-safe; lives here so workers never import the
#    Qt GUI module) ─────────────────────────────────────────────────────────
_W = {}  # per-worker context, populated by _worker_init


def _worker_init(ctx):
    """ProcessPool initializer: runs once per worker. Attaches the shared-memory
    cube, rebuilds the model + params + stellar prep, and pins BLAS to 1 thread
    so N workers don't oversubscribe the cores."""
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[_v] = '1'

    shm = shared_memory.SharedMemory(name=ctx['shm_name'])
    cube = np.ndarray(ctx['shape'], dtype=np.dtype(ctx['dtype']), buffer=shm.buf)

    params = Parameters()
    params.loads(ctx['params_dumps'])
    model = build_model(ctx['n_regions'], ctx['n_lines'])

    # Prepare a stellar template library per region spec (once per worker).
    preps = []
    if hcppxf is not None and ctx.get('stellar_specs'):
        wl, z, R = ctx['wavelengths'], ctx['z'], ctx['R']
        for spec in ctx['stellar_specs']:
            try:
                velscale, lam_rest = hcppxf.galaxy_velscale(wl, z, spec['fit_range'])
                lib = hcppxf.TemplateLibrary(spec['library']).load()
                lib.prepare(velscale, R, lam_rest)
                preps.append(dict(rid=spec['rid'], lib=lib, velscale=velscale,
                                  mask=ctx['stellar_mask'], moments=spec['moments'],
                                  fit_range=spec['fit_range'], library=spec['library']))
            except Exception as e:
                print(f"worker stellar prep failed ({spec.get('library')}): {e}")

    _W.update(shm=shm, cube=cube, params=params, model=model, df=ctx['df'],
              df_cont=ctx['df_cont'], wavelengths=ctx['wavelengths'], z=ctx['z'],
              R=ctx['R'], sequential=ctx['sequential'], max_nfev=ctx['max_nfev'],
              preps=preps)


def _worker_fit_one(task):
    """Fit one spaxel in a worker. task = (i, j, ra, dec). Returns
    (line_rows, stellar_rows) — both small, picklable lists of dicts."""
    i, j, ra, dec = task
    wl = _W['wavelengths']
    flux = np.nan_to_num(_W['cube'][:, j, i].astype(float))

    stellar_baseline = np.zeros_like(wl, dtype=float)
    stellar_kin, stellar_rows = {}, []
    for prep in _W['preps']:
        kin, base = fit_stellar_one(flux, wl, _W['z'], _W['R'], prep)
        if kin is not None:
            stellar_baseline = stellar_baseline + base
            stellar_kin[kin['region_ID']] = kin
            stellar_rows.append({
                'spaxel_x': i, 'spaxel_y': j, 'region_ID': kin['region_ID'],
                'RA': float(ra), 'Dec': float(dec),
                'stellar_V': kin['V'], 'stellar_sigma': kin['sigma'],
                'stellar_h3': kin['h3'], 'stellar_h4': kin['h4'],
                'stellar_scale': kin['scale'], 'stellar_chi2': kin['chi2'],
                'success': True})

    line_rows = fit_one_spaxel(
        flux, stellar_baseline, wl, _W['params'], _W['model'], _W['df'],
        _W['df_cont'], _W['z'], _W['max_nfev'], _W['sequential'], (i, j),
        ra, dec, stellar_kin)
    return line_rows, stellar_rows
