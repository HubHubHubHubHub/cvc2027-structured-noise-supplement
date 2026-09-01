#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expérience 2 : propagation calibrée de (A, Gamma) sur BrainWeb.

Le script est autonome, déterministe et n'écrit que dans son propre
répertoire. Les volumes et le JSON de l'expérience précédente sont lus
depuis le répertoire frère ``code_brainweb``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pathlib
import platform
import sys
import time
from datetime import datetime


HERE = pathlib.Path(__file__).resolve().parent
BRAINWEB_DIR = HERE.parent / "code_brainweb"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib-cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import chi2  # noqa: E402


SHAPE = (181, 217, 181)  # (z, y, x)
NY, NX = 217, 181
NPIX = NY * NX
T2 = np.array(
    [0.0, 1.0, 0.62, 0.45, 0.60, 0.55, 0.60, 0.15, 0.60, 0.50, 0.92]
)

FRAC_LINES = 0.25
CENTER_LINES = 8
SIGMA_NOISE = 0.02
PRIOR_LAMBDA = 25.0
PRIOR_EPS = 0.5
PRIOR_MEAN = 0.35
Z_CAL = (71, 81, 91, 111, 121)
Z_C0 = (76, 106)
Z_TEST = 101
J_CAL = 24
J_TEST = 400
B_BOOT = 500
RADIAL_BINS = 64
SPECTRAL_FLOOR_FACTOR = 1.0e-3
TEST_CHUNK = 40

SEEDS = {
    "mask": 7,
    "calibration_noise": 20260829,
    "test_noise": 314159,
    "bootstrap": 271828,
    "dense_internal": 161803,
}

KY = np.fft.fftfreq(NY)[:, None]
KX = np.fft.fftfreq(NX)[None, :]
KY_GRID = np.broadcast_to(KY, (NY, NX))
KX_GRID = np.broadcast_to(KX, (NY, NX))
W_AX = 2.0 * np.pi * np.abs(KY_GRID)
W_LAT = 2.0 * np.pi * np.abs(KX_GRID)
LAP = 4.0 - 2.0 * np.cos(2.0 * np.pi * KY) - 2.0 * np.cos(
    2.0 * np.pi * KX
)


def charge_volume(nom: str, dtype: str | np.dtype) -> np.ndarray:
    chemin = BRAINWEB_DIR / f"{nom}.rawb.gz"
    raw = gzip.decompress(chemin.read_bytes())
    attendu = int(np.prod(SHAPE)) * np.dtype(dtype).itemsize
    if len(raw) != attendu:
        raise ValueError(f"{chemin.name}: {len(raw)} octets, attendu {attendu}")
    return np.frombuffer(raw, dtype=dtype).reshape(SHAPE)


def empreinte(chemin: pathlib.Path) -> dict:
    contenu = chemin.read_bytes()
    return {
        "file": chemin.name,
        "compressed_bytes": len(contenu),
        "sha256": hashlib.sha256(contenu).hexdigest(),
    }


def empreinte_tableau(x: np.ndarray) -> str:
    canonique = np.ascontiguousarray(x, dtype="<f8")
    return hashlib.sha256(canonique.tobytes()).hexdigest()


def coupe(labels: np.ndarray, z: int) -> np.ndarray:
    if labels[z].max() >= len(T2):
        raise ValueError("Classe BrainWeb hors table d'intensités")
    return T2[labels[z]].astype(float)


def masque_lignes(
    frac: float, centre: int, rng: np.random.Generator
) -> np.ndarray:
    garde = np.zeros(NY, dtype=bool)
    garde[: centre + 1] = True
    garde[-centre:] = True
    autres = np.arange(centre + 1, NY - centre)
    moitie_positive = autres[autres <= NY // 2]
    selection = moitie_positive[rng.random(len(moitie_positive)) < frac]
    garde[selection] = True
    garde[(NY - selection) % NY] = True
    if not np.array_equal(garde, garde[(-np.arange(NY)) % NY]):
        raise AssertionError("Le masque de lignes n'est pas hermitien")
    return np.broadcast_to(garde[:, None], (NY, NX)).copy()


def gain_posterieur(masque: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phat = PRIOR_LAMBDA * LAP**2 + PRIOR_EPS
    q = masque / SIGMA_NOISE**2 + phat
    gain = masque.astype(float) / (masque.astype(float) + SIGMA_NOISE**2 * phat)
    return gain, q


def moyenne_sans_bruit(
    xstar: np.ndarray, masque: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    phat = PRIOR_LAMBDA * LAP**2 + PRIOR_EPS
    q = masque / SIGMA_NOISE**2 + phat
    xhat = np.fft.fft2(xstar, norm="ortho")
    xbarhat = np.fft.fft2(np.full_like(xstar, PRIOR_MEAN), norm="ortho")
    mhat = (masque * xhat / SIGMA_NOISE**2 + phat * xbarhat) / q
    m_complexe = np.fft.ifft2(mhat, norm="ortho")
    if float(np.max(np.abs(m_complexe.imag))) > 2.0e-12:
        raise AssertionError("La moyenne sans bruit n'est pas réelle")
    return m_complexe.real, mhat


def reconstruction_depuis_bruit(
    xstar: np.ndarray, masque: np.ndarray, eta: np.ndarray
) -> np.ndarray:
    """Route explicite de posterior(), avec un bruit fourni."""
    phat = PRIOR_LAMBDA * LAP**2 + PRIOR_EPS
    q = masque / SIGMA_NOISE**2 + phat
    xhat = np.fft.fft2(xstar, norm="ortho")
    y_sans_bruit = np.fft.ifft2(masque * xhat, norm="ortho").real
    yhat = np.fft.fft2(y_sans_bruit + eta, norm="ortho")
    xbarhat = np.fft.fft2(np.full_like(xstar, PRIOR_MEAN), norm="ortho")
    muhat = (masque * yhat / SIGMA_NOISE**2 + phat * xbarhat) / q
    return np.fft.ifft2(muhat, norm="ortho").real


def resume_biais(
    z: int, biais: np.ndarray, masque: np.ndarray
) -> dict:
    bhat = np.fft.fft2(biais, norm="ortho")
    return {
        "slice_z": int(z),
        "spatial_mean": float(np.mean(biais)),
        "spatial_rmse": float(np.sqrt(np.mean(biais**2))),
        "l2_norm": float(np.linalg.norm(biais)),
        "max_abs": float(np.max(np.abs(biais))),
        "fourier_energy_observed": float(np.sum(np.abs(bhat[masque]) ** 2)),
        "fourier_energy_unobserved": float(
            np.sum(np.abs(bhat[~masque]) ** 2)
        ),
        "field_sha256_float64": empreinte_tableau(biais),
    }


def calibre_et_prepare_c0(
    vol_normal: np.ndarray, masque: np.ndarray
) -> dict:
    debut = time.perf_counter()
    rng = np.random.default_rng(SEEDS["calibration_noise"])
    gain, _ = gain_posterieur(masque)
    puissances = np.empty((len(Z_CAL), J_CAL, NY, NX), dtype=np.float64)
    biais = []
    erreurs_fft_spatial = []

    for indice, z in enumerate(Z_CAL):
        xstar = coupe(vol_normal, z)
        m, _ = moyenne_sans_bruit(xstar, masque)
        eta = rng.normal(0.0, SIGMA_NOISE, size=(J_CAL, NY, NX))
        etahat = np.fft.fft2(eta, axes=(-2, -1), norm="ortho")
        zhat = gain[None, :, :] * etahat
        zch = zhat - zhat.mean(axis=0, keepdims=True)
        zch[:, ~masque] = 0.0
        puissances[indice] = np.abs(zch) ** 2
        champs = np.fft.ifft2(zhat, axes=(-2, -1), norm="ortho").real
        residus = (m - xstar)[None, :, :] + champs
        bhat_s = residus.mean(axis=0)
        biais.append(resume_biais(z, bhat_s, masque))
        spatial_centre = residus - bhat_s[None, :, :]
        fft_spatial = np.fft.fft2(
            spatial_centre, axes=(-2, -1), norm="ortho"
        )
        erreurs_fft_spatial.append(float(np.max(np.abs(fft_spatial - zch))))

    denominateur = len(Z_CAL) * (J_CAL - 1)
    khat = puissances.sum(axis=(0, 1)) / denominateur
    correction = J_CAL / (J_CAL - 1)
    khat_route_moyenne = correction * puissances.mean(axis=(0, 1))
    erreur_correction = float(np.max(np.abs(khat - khat_route_moyenne)))

    # Les deux coupes C0 suivent dans le même flux de bruit désigné.
    c0_centres_hat = np.empty((len(Z_C0), J_CAL, NY, NX), dtype=np.complex128)
    c0_biais = []
    for indice, z in enumerate(Z_C0):
        xstar = coupe(vol_normal, z)
        m, _ = moyenne_sans_bruit(xstar, masque)
        eta = rng.normal(0.0, SIGMA_NOISE, size=(J_CAL, NY, NX))
        etahat = np.fft.fft2(eta, axes=(-2, -1), norm="ortho")
        zhat = gain[None, :, :] * etahat
        centres = zhat - zhat.mean(axis=0, keepdims=True)
        # Un champ centré a marginalement (J-1)/J fois la covariance.
        # Cette remise à l'échelle rend la NLL C0 comparable à K_hat,
        # qui estime K_true aprè correction de Bessel.
        centres *= math.sqrt(J_CAL / (J_CAL - 1))
        centres[:, ~masque] = 0.0
        c0_centres_hat[indice] = centres
        champs = np.fft.ifft2(zhat, axes=(-2, -1), norm="ortho").real
        residus = (m - xstar)[None, :, :] + champs
        c0_biais.append(resume_biais(z, residus.mean(axis=0), masque))

    return {
        "khat": khat,
        "powers": puissances,
        "bias_summaries": biais,
        "c0_centered_hat": c0_centres_hat,
        "c0_bias_summaries": c0_biais,
        "centering_correction": correction,
        "centering_denominator": denominateur,
        "max_abs_khat_two_correction_routes": erreur_correction,
        "max_abs_fft_spatial_vs_direct_centered": float(max(erreurs_fft_spatial)),
        "max_abs_centered_coefficient_unobserved": float(
            np.max(np.abs(c0_centres_hat[:, :, ~masque]))
        ),
        "c0_centered_field_marginal_scale": float(
            math.sqrt(J_CAL / (J_CAL - 1))
        ),
        "duration_seconds": float(time.perf_counter() - debut),
    }


def quantile_ecart_chi2(probabilite: float, ddl: int) -> float:
    def fonction(ecart: float) -> float:
        bas = max(0.0, ddl * (1.0 - ecart))
        haut = ddl * (1.0 + ecart)
        return float(chi2.cdf(haut, ddl) - chi2.cdf(bas, ddl) - probabilite)

    return float(brentq(fonction, 0.0, 2.0))


def comparaison_ktrue(
    khat: np.ndarray, ktrue: np.ndarray, masque: np.ndarray
) -> dict:
    ratio = khat[masque] / ktrue[masque]
    ecart = np.abs(ratio - 1.0)
    resultat = {
        "observed_mode_count_full_spectrum": int(masque.sum()),
        "median_abs_relative_error": float(np.median(ecart)),
        "mean_ratio_khat_over_ktrue": float(np.mean(ratio)),
        "median_ratio_khat_over_ktrue": float(np.median(ratio)),
        "rmse_relative": float(np.sqrt(np.mean((ratio - 1.0) ** 2))),
        "max_khat_unobserved": float(np.max(khat[~masque])),
        "sum_khat_unobserved": float(np.sum(khat[~masque])),
    }
    for nom, ddl in (("designated_chi_square_120", 120), ("exact_generic_chi_square_230", 230)):
        intervalle = chi2.ppf([0.025, 0.975], ddl) / ddl
        resultat[nom] = {
            "degrees_of_freedom": ddl,
            "median_abs_relative_error_reference": quantile_ecart_chi2(0.5, ddl),
            "central_pointwise_95_ratio_interval": [
                float(intervalle[0]),
                float(intervalle[1]),
            ],
            "fraction_observed_ratios_inside_pointwise_95_interval": float(
                np.mean((ratio >= intervalle[0]) & (ratio <= intervalle[1]))
            ),
        }
    resultat["degrees_of_freedom_note"] = (
        "The mission-requested chi-square(120) benchmark is reported. "
        "After five within-slice mean removals, a generic complex Fourier "
        "mode has 5*(24-1)=115 complex degrees of freedom, hence the exact "
        "real chi-square benchmark is 230; DC instead has 115."
    )
    return resultat


def normalise_trace(
    spectre: np.ndarray, trace_cible: float
) -> tuple[np.ndarray, float]:
    s = np.maximum(np.asarray(spectre, dtype=float), 0.0)
    facteur = trace_cible / float(np.sum(s))
    return s * facteur, float(facteur)


def moyenne_radiale(khat: np.ndarray) -> tuple[np.ndarray, dict]:
    rayon = np.sqrt(KY_GRID**2 + KX_GRID**2)
    aretes = np.linspace(0.0, float(rayon.max()) + np.finfo(float).eps, RADIAL_BINS + 1)
    indices = np.clip(np.digitize(rayon.ravel(), aretes) - 1, 0, RADIAL_BINS - 1)
    sommes = np.bincount(indices, weights=khat.ravel(), minlength=RADIAL_BINS)
    comptes = np.bincount(indices, minlength=RADIAL_BINS)
    moyennes = np.divide(sommes, comptes, out=np.zeros_like(sommes), where=comptes > 0)
    centres = 0.5 * (aretes[:-1] + aretes[1:])
    valides = comptes > 0
    radial = np.interp(
        rayon.ravel(), centres[valides], moyennes[valides],
        left=float(moyennes[valides][0]), right=float(moyennes[valides][-1])
    ).reshape(NY, NX)
    return radial, {
        "n_annuli": RADIAL_BINS,
        "empty_annuli": int(np.sum(~valides)),
        "annulus_centers_cycles_per_pixel": [float(x) for x in centres],
        "annulus_means_before_trace_normalisation": [float(x) for x in moyennes],
    }


def symetrise_quadrants(khat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ay = np.arange(NY // 2 + 1, dtype=float) / NY
    ax = np.arange(NX // 2 + 1, dtype=float) / NX
    quadrant = np.empty((len(ay), len(ax)), dtype=float)
    for iy in range(len(ay)):
        ys = [0] if iy == 0 else [iy, (-iy) % NY]
        for ix in range(len(ax)):
            xs = [0] if ix == 0 else [ix, (-ix) % NX]
            quadrant[iy, ix] = float(np.mean(khat[np.ix_(ys, xs)]))
    return ay, ax, quadrant


def echange_axes_continu(khat: np.ndarray) -> tuple[np.ndarray, dict]:
    ay, ax, quadrant = symetrise_quadrants(khat)
    interpolateur = RegularGridInterpolator(
        (ay, ax), quadrant, method="linear", bounds_error=False, fill_value=None
    )
    cible_y = np.broadcast_to(np.abs(KX_GRID), (NY, NX))
    cible_x = np.broadcast_to(np.abs(KY_GRID), (NY, NX))
    points = np.column_stack((cible_y.ravel(), cible_x.ravel()))
    swap = interpolateur(points).reshape(NY, NX)
    minimum_avant = float(np.min(swap))
    swap = np.maximum(swap, 0.0)
    iy_inv = (-np.arange(NY)) % NY
    ix_inv = (-np.arange(NX)) % NX
    swap = 0.5 * (swap + swap[np.ix_(iy_inv, ix_inv)])
    hors = (points[:, 0] > ay[-1]) | (points[:, 1] > ax[-1])
    return swap, {
        "method": "RegularGridInterpolator on sign-averaged absolute-frequency quadrant",
        "source_grid_shape": [int(len(ay)), int(len(ax))],
        "bilinear_extrapolated_target_count_at_odd_grid_edge": int(np.sum(hors)),
        "minimum_before_nonnegative_clip": minimum_avant,
        "negative_values_clipped": int(np.sum(interpolateur(points) < 0.0)),
    }


def construit_geometries(khat: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    trace_cible = float(np.sum(khat))
    radial_brut, diag_radial = moyenne_radiale(khat)
    swap_brut, diag_swap = echange_axes_continu(khat)
    white_brut = np.ones_like(khat)
    bruts = {
        "K_cal": khat.copy(),
        "K_radial": radial_brut,
        "K_swap": swap_brut,
        "K_white": white_brut,
    }
    geoms = {}
    facteurs = {}
    for nom, brut in bruts.items():
        geoms[nom], facteurs[nom] = normalise_trace(brut, trace_cible)
    diagnostic = {
        "target_trace_sum_spectrum": trace_cible,
        "normalisation_factors": facteurs,
        "trace_after_normalisation": {
            nom: float(np.sum(k)) for nom, k in geoms.items()
        },
        "max_abs_trace_difference": float(
            max(abs(np.sum(k) - trace_cible) for k in geoms.values())
        ),
        "radial_construction": diag_radial,
        "swap_construction": diag_swap,
    }
    return geoms, diagnostic


def nll_spectrale_par_champ(
    champs_hat: np.ndarray, spectre: np.ndarray
) -> np.ndarray:
    if np.any(spectre <= 0.0):
        raise ValueError("Spectre non strictement positif dans la NLL")
    puissance = np.abs(champs_hat) ** 2
    constante = np.log(2.0 * np.pi * spectre)
    return 0.5 * np.sum(
        constante[None, None, :, :] + puissance / spectre[None, None, :, :],
        axis=(-2, -1),
    )


def bootstrap_stratifie_moyenne(
    valeurs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    # valeurs : strates x observations
    sorties = np.empty(B_BOOT, dtype=float)
    for b in range(B_BOOT):
        morceaux = []
        for s in range(valeurs.shape[0]):
            indices = rng.integers(0, valeurs.shape[1], size=valeurs.shape[1])
            morceaux.append(valeurs[s, indices])
        sorties[b] = float(np.mean(np.concatenate(morceaux)))
    return sorties


def evalue_c0(
    c0_hat: np.ndarray,
    geoms: dict[str, np.ndarray],
    delta2: float,
    masque: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    debut = time.perf_counter()
    nll_cal = nll_spectrale_par_champ(c0_hat, geoms["K_cal"] + delta2)
    nll_swap = nll_spectrale_par_champ(c0_hat, geoms["K_swap"] + delta2)
    differences = nll_swap - nll_cal
    spec_cal = geoms["K_cal"] + delta2
    spec_swap = geoms["K_swap"] + delta2
    puissance_moyenne = np.mean(np.abs(c0_hat) ** 2, axis=(0, 1))
    terme_log_mode = 0.5 * np.log(spec_swap / spec_cal)
    terme_quadratique_mode = 0.5 * puissance_moyenne * (
        1.0 / spec_swap - 1.0 / spec_cal
    )

    def decomposition(support: np.ndarray) -> dict:
        log = float(np.sum(terme_log_mode[support]))
        quad = float(np.sum(terme_quadratique_mode[support]))
        return {
            "log_determinant_contribution_per_field": log,
            "quadratic_contribution_at_mean_periodogram_per_field": quad,
            "total_contribution_per_field": float(log + quad),
            "mode_count": int(np.sum(support)),
        }

    boot = bootstrap_stratifie_moyenne(differences, rng)
    moyenne = float(np.mean(differences))
    intervalle = np.quantile(boot, [0.025, 0.975])
    passe = bool(moyenne > 0.0 and intervalle[0] > 0.0)
    return {
        "name": "C0_identifiability_on_held_out_normal_slices",
        "held_out_slices": [int(z) for z in Z_C0],
        "n_centered_fields": int(differences.size),
        "statistic": "NLL(K_swap) - NLL(K_cal)",
        "mean_difference_per_field": moyenne,
        "total_difference_48_fields": float(np.sum(differences)),
        "paired_bootstrap_95_interval_mean_per_field": [
            float(intervalle[0]), float(intervalle[1])
        ],
        "bootstrap_standard_error_mean": float(np.std(boot, ddof=1)),
        "mean_nll_K_cal_per_field": float(np.mean(nll_cal)),
        "mean_nll_K_swap_per_field": float(np.mean(nll_swap)),
        "difference_decomposition": {
            "observed_modes": decomposition(masque),
            "unobserved_modes": decomposition(~masque),
            "all_modes": decomposition(np.ones_like(masque, dtype=bool)),
            "reconstruction_abs_error_vs_direct_mean_difference": float(
                abs(np.sum(terme_log_mode + terme_quadratique_mode) - moyenne)
            ),
            "reconstruction_relative_error_vs_direct_mean_difference": float(
                abs(np.sum(terme_log_mode + terme_quadratique_mode) - moyenne)
                / abs(moyenne)
            ),
            "observed_modes_Kswap_at_most_10_times_floor": int(
                np.sum(masque & (geoms["K_swap"] <= 10.0 * delta2))
            ),
            "observed_modes_Kswap_at_most_1000_times_floor": int(
                np.sum(masque & (geoms["K_swap"] <= 1000.0 * delta2))
            ),
            "median_Kcal_on_observed_modes": float(np.median(geoms["K_cal"][masque])),
            "median_Kswap_on_observed_modes": float(np.median(geoms["K_swap"][masque])),
            "explanation": (
                "Axis exchange moves the ky-line support toward kx. Held-out "
                "centered residuals retain power only on the fixed observed ky "
                "lines, so the quadratic NLL term is enormous wherever K_swap "
                "is zero and only the declared common delta2 remains."
            ),
        },
        "pass": passe,
        "duration_seconds": float(time.perf_counter() - debut),
    }


def construit_observables(vol_ms: np.ndarray) -> dict:
    labels = vol_ms[Z_TEST]
    lesions = labels == 10
    roi = binary_dilation(lesions, iterations=2)
    anneau = binary_dilation(lesions, iterations=5) & ~roi & (labels == 3)
    if not lesions.any() or not anneau.any():
        raise ValueError("ROI lésionnelle ou anneau de substance blanche vide")
    g = np.zeros((NY, NX), dtype=float)
    g[lesions] = 1.0 / int(lesions.sum())
    g[anneau] = -1.0 / int(anneau.sum())
    return {
        "lesions": lesions,
        "roi": roi,
        "wm_ring": anneau,
        "g": g,
        "w": anneau.astype(float),
    }


def applique_d(x: np.ndarray, axis: int) -> np.ndarray:
    return np.roll(x, -1, axis=axis) - x


def applique_dt(x: np.ndarray, axis: int) -> np.ndarray:
    return np.roll(x, 1, axis=axis) - x


def applique_b(x: np.ndarray, w: np.ndarray, axis: int) -> np.ndarray:
    return applique_dt(w * applique_d(x, axis), axis)


def valeur_q(x: np.ndarray, w: np.ndarray, axis: int) -> np.ndarray:
    dx = applique_d(x, axis)
    return np.sum(w * dx**2, axis=(-2, -1))


def trace_b1_k_b2_k(
    k: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    noyau_fft: np.ndarray,
) -> float:
    a = k * np.conjugate(d1) * d2
    convolution = np.fft.ifft2(
        noyau_fft * np.fft.fft2(np.conjugate(a))
    )
    valeur = np.sum(a * convolution) / a.size
    if abs(float(valeur.imag)) > 2.0e-9 * max(1.0, abs(float(valeur.real))):
        raise AssertionError("Trace croisée complexe")
    return float(valeur.real)


def prepare_moments(m: np.ndarray, obs: dict) -> dict:
    d_ax = np.broadcast_to(np.exp(2j * np.pi * KY) - 1.0, (NY, NX)).copy()
    d_lat = np.broadcast_to(np.exp(2j * np.pi * KX) - 1.0, (NY, NX)).copy()
    bm_ax = applique_b(m, obs["w"], axis=0)
    bm_lat = applique_b(m, obs["w"], axis=1)
    what = np.fft.fft2(obs["w"], norm="ortho")
    noyau = np.abs(what) ** 2
    return {
        "m": m,
        "w": obs["w"],
        "w_count": float(np.sum(obs["w"])),
        "g": obs["g"],
        "ghat": np.fft.fft2(obs["g"], norm="ortho"),
        "d_ax": d_ax,
        "d_lat": d_lat,
        "absd2_ax": np.abs(d_ax) ** 2,
        "absd2_lat": np.abs(d_lat) ** 2,
        "bm_ax_hat": np.fft.fft2(bm_ax, norm="ortho"),
        "bm_lat_hat": np.fft.fft2(bm_lat, norm="ortho"),
        "kernel_fft": np.fft.fft2(noyau),
        "base_means": np.array(
            [
                float(np.sum(obs["g"] * m)),
                float(valeur_q(m, obs["w"], axis=0)),
                float(valeur_q(m, obs["w"], axis=1)),
            ]
        ),
    }


def prediction_moments(k: np.ndarray, prep: dict) -> dict:
    ghat = prep["ghat"]
    bmh_a = prep["bm_ax_hat"]
    bmh_l = prep["bm_lat_hat"]
    d_a = prep["d_ax"]
    d_l = prep["d_lat"]
    abs_a = prep["absd2_ax"]
    abs_l = prep["absd2_lat"]
    n = k.size
    wcount = prep["w_count"]

    var_f = float(np.sum(k * np.abs(ghat) ** 2))
    shift_a = float(wcount / n * np.sum(abs_a * k))
    shift_l = float(wcount / n * np.sum(abs_l * k))
    main_aa = float(np.sum(k * np.abs(bmh_a) ** 2))
    main_ll = float(np.sum(k * np.abs(bmh_l) ** 2))
    main_al = float(np.real(np.sum(k * np.conjugate(bmh_a) * bmh_l)))
    trace_aa = trace_b1_k_b2_k(k, d_a, d_a, prep["kernel_fft"])
    trace_ll = trace_b1_k_b2_k(k, d_l, d_l, prep["kernel_fft"])
    trace_al = trace_b1_k_b2_k(k, d_a, d_l, prep["kernel_fft"])
    var_a = 4.0 * main_aa + 2.0 * trace_aa
    var_l = 4.0 * main_ll + 2.0 * trace_ll
    cov_fa = 2.0 * float(
        np.real(np.sum(np.conjugate(ghat) * k * bmh_a))
    )
    cov_fl = 2.0 * float(
        np.real(np.sum(np.conjugate(ghat) * k * bmh_l))
    )
    cov_al = 4.0 * main_al + 2.0 * trace_al
    covariance = np.array(
        [[var_f, cov_fa, cov_fl], [cov_fa, var_a, cov_al], [cov_fl, cov_al, var_l]]
    )
    deplacements = np.array([0.0, shift_a, shift_l])
    return {
        "mean_shifts": deplacements,
        "means": prep["base_means"] + deplacements,
        "variances": np.diag(covariance).copy(),
        "covariance": covariance,
        "components": {
            "m_Bax_K_Bax_m": main_aa,
            "m_Blat_K_Blat_m": main_ll,
            "m_Bax_K_Blat_m": main_al,
            "trace_BaxK_squared": trace_aa,
            "trace_BlatK_squared": trace_ll,
            "trace_BaxK_BlatK": trace_al,
        },
    }


def erreur_relative(a: float, b: float) -> float:
    return float(abs(a - b) / max(abs(b), 1.0e-30))


def test_dense_interne() -> dict:
    debut = time.perf_counter()
    n = 8
    n2 = n * n
    rng = np.random.default_rng(SEEDS["dense_internal"])
    brut = 0.2 + rng.random((n, n))
    inv = (-np.arange(n)) % n
    k = 0.5 * (brut + brut[np.ix_(inv, inv)])

    fmat = np.empty((n2, n2), dtype=np.complex128)
    for j in range(n2):
        base = np.zeros((n, n), dtype=float)
        base.ravel()[j] = 1.0
        fmat[:, j] = np.fft.fft2(base, norm="ortho").ravel()
    kdense_complexe = fmat.conj().T @ np.diag(k.ravel()) @ fmat
    imag_k = float(np.max(np.abs(kdense_complexe.imag)))
    kdense = kdense_complexe.real

    def d_dense(axis: int) -> np.ndarray:
        d = np.empty((n2, n2), dtype=float)
        for j in range(n2):
            base = np.zeros((n, n), dtype=float)
            base.ravel()[j] = 1.0
            d[:, j] = (np.roll(base, -1, axis=axis) - base).ravel()
        return d

    da = d_dense(0)
    dl = d_dense(1)
    w = (rng.random((n, n)) > 0.57).astype(float)
    if w.sum() == 0:
        raise AssertionError("W dense vide")
    wdense = np.diag(w.ravel())
    ba = da.T @ wdense @ da
    bl = dl.T @ wdense @ dl
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    dsa = np.broadcast_to(np.exp(2j * np.pi * fy) - 1.0, (n, n)).copy()
    dsl = np.broadcast_to(np.exp(2j * np.pi * fx) - 1.0, (n, n)).copy()
    what = np.fft.fft2(w, norm="ortho")
    noyau_fft = np.fft.fft2(np.abs(what) ** 2)
    x = rng.normal(size=n2)
    g = rng.normal(size=n2)
    m = rng.normal(size=n2)

    bxa = ba @ x
    lhs_g = float(g @ kdense @ g)
    rhs_g = float(np.sum(k * np.abs(np.fft.fft2(g.reshape(n, n), norm="ortho")) ** 2))
    lhs_bx = float(bxa @ kdense @ bxa)
    rhs_bx = float(np.sum(k * np.abs(np.fft.fft2(bxa.reshape(n, n), norm="ortho")) ** 2))
    lhs_trace_a = float(np.trace(ba @ kdense))
    rhs_trace_a = float(w.sum() / n2 * np.sum(np.abs(dsa) ** 2 * k))
    lhs_trace_l = float(np.trace(bl @ kdense))
    rhs_trace_l = float(w.sum() / n2 * np.sum(np.abs(dsl) ** 2 * k))
    lhs_sq_a = float(np.trace(ba @ kdense @ ba @ kdense))
    rhs_sq_a = trace_b1_k_b2_k(k, dsa, dsa, noyau_fft)
    lhs_sq_l = float(np.trace(bl @ kdense @ bl @ kdense))
    rhs_sq_l = trace_b1_k_b2_k(k, dsl, dsl, noyau_fft)
    lhs_cross = float(np.trace(ba @ kdense @ bl @ kdense))
    rhs_cross = trace_b1_k_b2_k(k, dsa, dsl, noyau_fft)

    bma = ba @ m
    bml = bl @ m
    varqa_dense = float(4.0 * bma @ kdense @ bma + 2.0 * lhs_sq_a)
    varql_dense = float(4.0 * bml @ kdense @ bml + 2.0 * lhs_sq_l)
    covfqa_dense = float(2.0 * g @ kdense @ bma)
    covfql_dense = float(2.0 * g @ kdense @ bml)
    covqal_dense = float(4.0 * bma @ kdense @ bml + 2.0 * lhs_cross)
    bmah = np.fft.fft2(bma.reshape(n, n), norm="ortho")
    bmlh = np.fft.fft2(bml.reshape(n, n), norm="ortho")
    gh = np.fft.fft2(g.reshape(n, n), norm="ortho")
    varqa_fft = float(4.0 * np.sum(k * np.abs(bmah) ** 2) + 2.0 * rhs_sq_a)
    varql_fft = float(4.0 * np.sum(k * np.abs(bmlh) ** 2) + 2.0 * rhs_sq_l)
    covfqa_fft = float(2.0 * np.real(np.sum(np.conjugate(gh) * k * bmah)))
    covfql_fft = float(2.0 * np.real(np.sum(np.conjugate(gh) * k * bmlh)))
    covqal_fft = float(
        4.0 * np.real(np.sum(k * np.conjugate(bmah) * bmlh)) + 2.0 * rhs_cross
    )

    identites = {
        "gT_K_g": erreur_relative(rhs_g, lhs_g),
        "xT_B_K_B_x": erreur_relative(rhs_bx, lhs_bx),
        "trace_Bax_K": erreur_relative(rhs_trace_a, lhs_trace_a),
        "trace_Blat_K": erreur_relative(rhs_trace_l, lhs_trace_l),
        "trace_BaxK_squared": erreur_relative(rhs_sq_a, lhs_sq_a),
        "trace_BlatK_squared": erreur_relative(rhs_sq_l, lhs_sq_l),
        "trace_BaxK_BlatK": erreur_relative(rhs_cross, lhs_cross),
        "var_Qax_closed_vs_dense": erreur_relative(varqa_fft, varqa_dense),
        "var_Qlat_closed_vs_dense": erreur_relative(varql_fft, varql_dense),
        "cov_F_Qax_closed_vs_dense": erreur_relative(covfqa_fft, covfqa_dense),
        "cov_F_Qlat_closed_vs_dense": erreur_relative(covfql_fft, covfql_dense),
        "cov_Qax_Qlat_closed_vs_dense": erreur_relative(covqal_fft, covqal_dense),
    }

    # Monte-Carlo dense, en lots : exactement 200 000 tirages.
    nmc = 200_000
    chol = np.linalg.cholesky(kdense)
    valeurs = np.empty((nmc, 3), dtype=float)
    position = 0
    while position < nmc:
        taille = min(10_000, nmc - position)
        z = rng.normal(size=(taille, n2)) @ chol.T
        xx = m[None, :] + z
        valeurs[position : position + taille, 0] = xx @ g
        valeurs[position : position + taille, 1] = np.einsum(
            "bi,ij,bj->b", xx, ba, xx, optimize=True
        )
        valeurs[position : position + taille, 2] = np.einsum(
            "bi,ij,bj->b", xx, bl, xx, optimize=True
        )
        position += taille
    covariance_mc = np.cov(valeurs, rowvar=False, ddof=1)
    centre = valeurs - valeurs.mean(axis=0)
    exacts = {
        "var_Qax": varqa_dense,
        "var_Qlat": varql_dense,
        "cov_F_Qax": covfqa_dense,
        "cov_F_Qlat": covfql_dense,
        "cov_Qax_Qlat": covqal_dense,
    }
    estimations = {
        "var_Qax": float(covariance_mc[1, 1]),
        "var_Qlat": float(covariance_mc[2, 2]),
        "cov_F_Qax": float(covariance_mc[0, 1]),
        "cov_F_Qlat": float(covariance_mc[0, 2]),
        "cov_Qax_Qlat": float(covariance_mc[1, 2]),
    }
    paires = {
        "var_Qax": (1, 1),
        "var_Qlat": (2, 2),
        "cov_F_Qax": (0, 1),
        "cov_F_Qlat": (0, 2),
        "cov_Qax_Qlat": (1, 2),
    }
    mc = {}
    for nom, (i, j) in paires.items():
        influence = centre[:, i] * centre[:, j]
        se = float(np.std(influence, ddof=1) / math.sqrt(nmc))
        zscore = float(abs(estimations[nom] - exacts[nom]) / se)
        mc[nom] = {
            "closed_form": float(exacts[nom]),
            "monte_carlo": float(estimations[nom]),
            "monte_carlo_standard_error": se,
            "absolute_discrepancy_in_standard_errors": zscore,
            "pass_below_3_standard_errors": bool(zscore < 3.0),
        }

    seuil_identites = 1.0e-10
    passe_identites = bool(max(identites.values()) < seuil_identites)
    passe_mc = bool(all(x["pass_below_3_standard_errors"] for x in mc.values()))
    return {
        "shape": [n, n],
        "random_spectrum_sha256_float64": empreinte_tableau(k),
        "max_imaginary_dense_K": imag_k,
        "relative_errors_fft_identity_vs_dense": identites,
        "maximum_relative_error": float(max(identites.values())),
        "relative_error_threshold": seuil_identites,
        "n_monte_carlo": nmc,
        "monte_carlo_comparisons": mc,
        "maximum_monte_carlo_discrepancy_in_standard_errors": float(
            max(x["absolute_discrepancy_in_standard_errors"] for x in mc.values())
        ),
        "pass": bool(passe_identites and passe_mc),
        "duration_seconds": float(time.perf_counter() - debut),
    }


def simule_test(
    xstar: np.ndarray, masque: np.ndarray, m: np.ndarray, obs: dict
) -> tuple[np.ndarray, dict]:
    debut = time.perf_counter()
    gain, _ = gain_posterieur(masque)
    rng = np.random.default_rng(SEEDS["test_noise"])
    valeurs = np.empty((J_TEST, 3), dtype=float)
    controles = []
    position = 0
    while position < J_TEST:
        taille = min(TEST_CHUNK, J_TEST - position)
        eta = rng.normal(0.0, SIGMA_NOISE, size=(taille, NY, NX))
        etahat = np.fft.fft2(eta, axes=(-2, -1), norm="ortho")
        zhat = gain[None, :, :] * etahat
        zcomplexe = np.fft.ifft2(zhat, axes=(-2, -1), norm="ortho")
        if float(np.max(np.abs(zcomplexe.imag))) > 2.0e-12:
            raise AssertionError("Bruit reconstruit complexe")
        champs = m[None, :, :] + zcomplexe.real
        valeurs[position : position + taille, 0] = np.einsum(
            "bij,ij->b", champs, obs["g"]
        )
        valeurs[position : position + taille, 1] = valeur_q(
            champs, obs["w"], axis=1
        )
        valeurs[position : position + taille, 2] = valeur_q(
            champs, obs["w"], axis=2
        )
        for local in range(min(taille, max(0, 3 - position))):
            explicite = reconstruction_depuis_bruit(xstar, masque, eta[local])
            image_lineaire = zcomplexe.real[local]
            controles.append(
                {
                    "repetition": int(position + local),
                    "max_abs_reconstruction_minus_m_minus_linear_noise": float(
                        np.max(np.abs((explicite - m) - image_lineaire))
                    ),
                    "max_abs_vectorised_vs_explicit_reconstruction": float(
                        np.max(np.abs(champs[local] - explicite))
                    ),
                }
            )
        position += taille
    return valeurs, {
        "three_exact_linearity_checks": controles,
        "max_linearity_error": float(
            max(x["max_abs_reconstruction_minus_m_minus_linear_noise"] for x in controles)
        ),
        "duration_seconds": float(time.perf_counter() - debut),
    }


def bootstrap_calibration_predictions(
    powers: np.ndarray, prep: dict, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    debut = time.perf_counter()
    moyens = np.empty((B_BOOT, 3), dtype=float)
    variances = np.empty((B_BOOT, 3), dtype=float)
    denominateur = powers.shape[0] * (powers.shape[1] - 1)
    probabilites = np.full(powers.shape[1], 1.0 / powers.shape[1])
    for b in range(B_BOOT):
        comptes = np.stack(
            [rng.multinomial(powers.shape[1], probabilites) for _ in range(powers.shape[0])]
        )
        kb = np.einsum("sj,sjyx->yx", comptes, powers, optimize=True) / denominateur
        pred = prediction_moments(kb, prep)
        moyens[b] = pred["mean_shifts"]
        variances[b] = pred["variances"]
    return moyens, variances, float(time.perf_counter() - debut)


def bootstrap_test_observables(
    valeurs: np.ndarray, base: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    debut = time.perf_counter()
    indices = rng.integers(0, len(valeurs), size=(B_BOOT, len(valeurs)))
    echantillons = valeurs[indices]
    moyens = echantillons.mean(axis=1) - base[None, :]
    variances = echantillons.var(axis=1, ddof=1)
    return moyens, variances, float(time.perf_counter() - debut)


def resume_bootstrap(
    noms: list[str], estime: np.ndarray, boot: np.ndarray
) -> dict:
    resultat = {}
    for i, nom in enumerate(noms):
        q = np.quantile(boot[:, i], [0.025, 0.975])
        resultat[nom] = {
            "estimate": float(estime[i]),
            "bootstrap_mean": float(np.mean(boot[:, i])),
            "bootstrap_standard_error": float(np.std(boot[:, i], ddof=1)),
            "bootstrap_percentile_95_interval": [float(q[0]), float(q[1])],
        }
    return resultat


def criteres_c1_c2(
    pred_mean: np.ndarray,
    pred_var: np.ndarray,
    emp_mean: np.ndarray,
    emp_var: np.ndarray,
    pred_mean_boot: np.ndarray,
    pred_var_boot: np.ndarray,
    emp_mean_boot: np.ndarray,
    emp_var_boot: np.ndarray,
) -> tuple[dict, dict, dict]:
    noms = [
        "mean_shift_Fc", "mean_shift_Qax", "mean_shift_Qlat",
        "variance_Fc", "variance_Qax", "variance_Qlat",
    ]
    pred = np.concatenate((pred_mean, pred_var))
    emp = np.concatenate((emp_mean, emp_var))
    pred_boot = np.concatenate((pred_mean_boot, pred_var_boot), axis=1)
    emp_boot = np.concatenate((emp_mean_boot, emp_var_boot), axis=1)
    se_pred = pred_boot.std(axis=0, ddof=1)
    se_emp = emp_boot.std(axis=0, ddof=1)
    denom = np.sqrt(se_pred**2 + se_emp**2)
    if np.any(denom <= 0.0):
        raise AssertionError("Erreur-type combinée nulle")
    z = (pred - emp) / denom
    zboot = (
        (pred_boot - pred[None, :]) - (emp_boot - emp[None, :])
    ) / denom[None, :]
    maxboot = np.max(np.abs(zboot), axis=1)
    seuil = float(np.quantile(maxboot, 0.95))
    details = {
        noms[i]: {
            "prediction": float(pred[i]),
            "empirical": float(emp[i]),
            "prediction_bootstrap_se": float(se_pred[i]),
            "empirical_bootstrap_se": float(se_emp[i]),
            "combined_se": float(denom[i]),
            "z": float(z[i]),
            "absolute_z": float(abs(z[i])),
        }
        for i in range(len(noms))
    }
    simultane = {
        "construction": (
            "95th percentile of max absolute centered bootstrap z over all "
            "six mean/variance targets; calibration and test replicates paired by index"
        ),
        "threshold_95": seuil,
        "bootstrap_max_abs_z_median": float(np.median(maxboot)),
        "bootstrap_replicates": B_BOOT,
        "details": details,
    }
    c1_indices = [3, 4, 5]
    c2_indices = [0, 1, 2]
    c1_max = float(np.max(np.abs(z[c1_indices])))
    c2_max = float(np.max(np.abs(z[c2_indices])))
    c1 = {
        "name": "C1_Gamma_predicts_variances",
        "simultaneous_threshold_95": seuil,
        "maximum_absolute_z": c1_max,
        "pass": bool(c1_max <= seuil),
        "targets": {noms[i]: details[noms[i]] for i in c1_indices},
    }
    c2 = {
        "name": "C2_A_predicts_mean_shifts",
        "simultaneous_threshold_95": seuil,
        "maximum_absolute_z": c2_max,
        "pass": bool(c2_max <= seuil),
        "targets": {noms[i]: details[noms[i]] for i in c2_indices},
    }
    return c1, c2, simultane


def nll_gaussienne_vecteur(
    valeurs: np.ndarray, moyenne: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, dict]:
    covariance = 0.5 * (covariance + covariance.T)
    signe, logdet = np.linalg.slogdet(covariance)
    eig = np.linalg.eigvalsh(covariance)
    if signe <= 0 or eig[0] <= 0.0:
        raise AssertionError("Covariance prédite non définie positive")
    diff = valeurs - moyenne[None, :]
    solution = np.linalg.solve(covariance, diff.T).T
    quad = np.einsum("bi,bi->b", diff, solution)
    nll = 0.5 * (3.0 * np.log(2.0 * np.pi) + logdet + quad)
    return nll, {
        "log_determinant": float(logdet),
        "eigenvalues": [float(x) for x in eig],
        "condition_number": float(eig[-1] / eig[0]),
    }


def evalue_c3(
    valeurs: np.ndarray,
    geometries_avec_oracle: dict[str, np.ndarray],
    delta2: float,
    prep: dict,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, dict]]:
    debut = time.perf_counter()
    scores = {}
    par_rep = {}
    for nom, k in geometries_avec_oracle.items():
        pred = prediction_moments(k + delta2, prep)
        nll, diag = nll_gaussienne_vecteur(valeurs, pred["means"], pred["covariance"])
        par_rep[nom] = nll
        scores[nom] = {
            "total_score": float(np.sum(nll)),
            "mean_score_per_repetition": float(np.mean(nll)),
            "predicted_means_with_spectral_floor": [float(x) for x in pred["means"]],
            "predicted_covariance_with_spectral_floor": [
                [float(x) for x in ligne] for ligne in pred["covariance"]
            ],
            **diag,
        }
    difference = par_rep["K_swap"] - par_rep["K_cal"]
    indices = rng.integers(0, J_TEST, size=(B_BOOT, J_TEST))
    boot = difference[indices].mean(axis=1)
    intervalle = np.quantile(boot, [0.025, 0.975])
    moyenne = float(np.mean(difference))
    critere = {
        "name": "C3_proper_law_score_causal_control",
        "statistic": "S(K_swap) - S(K_cal)",
        "difference_total": float(np.sum(difference)),
        "difference_mean_per_repetition": moyenne,
        "paired_bootstrap_95_interval_mean_per_repetition": [
            float(intervalle[0]), float(intervalle[1])
        ],
        "paired_bootstrap_standard_error": float(np.std(boot, ddof=1)),
        "pass": bool(moyenne > 0.0 and intervalle[0] > 0.0),
        "duration_seconds": float(time.perf_counter() - debut),
    }
    return critere, scores


def modele_matern_ancien(
    s2: float, l_ax: float, l_lat: float, nu: float
) -> np.ndarray:
    return s2 * (1.0 + l_ax**2 * W_AX**2 + l_lat**2 * W_LAT**2) ** (-nu)


def normalise_ancien(spectre: np.ndarray) -> np.ndarray:
    s = np.broadcast_to(spectre, (NY, NX)).astype(float, copy=True)
    s += 1.0e-6 * float(s.mean())
    return s * NPIX / float(s.sum())


def distances_loi_sampler(vol_ms: np.ndarray) -> dict:
    debut = time.perf_counter()
    ancien = json.loads((BRAINWEB_DIR / "ms_results.json").read_text(encoding="utf-8"))
    theta = ancien["calibration"]["theta_hat"]
    ajuste = modele_matern_ancien(**theta)
    echange = modele_matern_ancien(
        theta["s2"], theta["l_lat"], theta["l_ax"], theta["nu"]
    )
    identite = np.ones((NY, NX), dtype=float)
    geoms = {
        "identity": identite,
        "calibrated": normalise_ancien(ajuste),
        "isotropised": identite.copy(),
        "axes-swapped": normalise_ancien(echange),
    }
    masque = masque_lignes(FRAC_LINES, CENTER_LINES, np.random.default_rng(7))
    xstar = coupe(vol_ms, Z_TEST)
    eta = np.random.default_rng(8).normal(0.0, SIGMA_NOISE, size=(NY, NX))
    mu = reconstruction_depuis_bruit(xstar, masque, eta)
    muhat = np.fft.fft2(mu, norm="ortho")
    x0hat = np.fft.fft2(np.full_like(xstar, PRIOR_MEAN), norm="ortho")
    _, q = gain_posterieur(masque)
    cible = 1.0 / q
    trace_cible = float(np.sum(cible))
    table = {}
    budget = 60
    facteur_pas = 1.8
    for nom, a in geoms.items():
        gamma = facteur_pas / float(np.max(a * q))
        rho = 1.0 - 0.5 * gamma * a * q
        vk_inf = gamma * a / (1.0 - rho**2)
        vk = vk_inf * (1.0 - rho ** (2 * budget))
        biais_hat = (x0hat - muhat) * rho**budget
        kl_mean = 0.5 * float(np.sum(q * np.abs(biais_hat) ** 2))
        ratio = q * vk
        kl_cov = 0.5 * float(np.sum(ratio - np.log(ratio) - 1.0))
        w2_mean = float(np.sum(np.abs(biais_hat) ** 2))
        w2_cov = float(np.sum((np.sqrt(vk) - np.sqrt(cible)) ** 2))
        table[nom] = {
            "gamma": float(gamma),
            "rho_min": float(np.min(rho)),
            "rho_max": float(np.max(rho)),
            "trace_V_K": float(np.sum(vk)),
            "posterior_trace": trace_cible,
            "fraction_posterior_variance_acquired": float(np.sum(vk) / trace_cible),
            "KL_mean_component": kl_mean,
            "KL_covariance_component": kl_cov,
            "KL_complete": float(kl_mean + kl_cov),
            "W2_squared_mean_component": w2_mean,
            "W2_squared_covariance_component": w2_cov,
            "W2_squared_complete": float(w2_mean + w2_cov),
        }
    identique = bool(np.array_equal(geoms["identity"], geoms["isotropised"]))
    egalites = {
        cle: bool(table["identity"][cle] == table["isotropised"][cle])
        for cle in table["identity"]
    }
    return {
        "source_parameters_file": "../code_brainweb/ms_results.json",
        "theta_read_from_source": theta,
        "budget_K": budget,
        "step_factor": facteur_pas,
        "route": "closed AR(1) law, recalculated in this script",
        "posterior_trace": trace_cible,
        "geometries": table,
        "identity_isotropised_proof": {
            "spectra_array_equal": identique,
            "all_reported_scalar_outputs_bitwise_equal": bool(all(egalites.values())),
            "per_output_equalities": egalites,
            "code_fact": "isotropised is constructed as identity.copy() in the inherited geometry",
        },
        "duration_seconds": float(time.perf_counter() - debut),
    }


def fait_figure(
    xstar: np.ndarray,
    obs: dict,
    khat: np.ndarray,
    kradial: np.ndarray,
    delta2: float,
    valeurs: np.ndarray,
    predictions: dict[str, dict],
    pred_mean_boot: np.ndarray,
    emp_mean_boot: np.ndarray,
    prep: dict,
) -> float:
    debut = time.perf_counter()
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    ax.imshow(xstar, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.contour(obs["roi"].astype(float), levels=[0.5], colors=["#f2c14e"], linewidths=0.8)
    ax.contour(obs["wm_ring"].astype(float), levels=[0.5], colors=["#00b4d8"], linewidths=0.8)
    ax.contour(obs["lesions"].astype(float), levels=[0.5], colors=["#d00000"], linewidths=1.0)
    ax.set_title("(i) Held-out MS slice z=101 and supports")
    ax.set_axis_off()
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="#d00000", label="lesions"),
            Patch(facecolor="none", edgecolor="#f2c14e", label="dilated ROI (2)"),
            Patch(facecolor="none", edgecolor="#00b4d8", label="WM ring"),
        ],
        loc="lower right", fontsize=8, framealpha=0.9,
    )

    ax = axes[0, 1]
    logk = np.log10(np.fft.fftshift(khat + delta2))
    image_k = ax.imshow(logk, cmap="magma", origin="lower", aspect="auto")
    ax.set_title("(ii) log10 of the centred calibrated spectrum")
    ax.set_xlabel("centred lateral frequency")
    ax.set_ylabel("centred axial frequency")
    fig.colorbar(image_k, ax=ax, shrink=0.82, label="log10 modal variance")
    ins = inset_axes(ax, width="29%", height="31%", loc="lower right", borderpad=1.1)
    ins.imshow(
        np.log10(np.fft.fftshift(kradial + delta2)),
        cmap="magma", origin="lower", aspect="auto",
        vmin=float(np.min(logk)), vmax=float(np.max(logk)),
    )
    ins.set_title("radial", fontsize=8)
    ins.set_xticks([])
    ins.set_yticks([])

    ax = axes[1, 0]
    couleurs = {
        "K_cal": "#0077b6", "K_radial": "#2a9d8f", "K_swap": "#e76f51",
        "K_white": "#6c757d", "K_true": "#6a4c93",
    }
    marqueurs = {"Fc": "o", "Qax": "s", "Qlat": "^"}
    emp_var = valeurs.var(axis=0, ddof=1)
    for geom, pred in predictions.items():
        for i, nom in enumerate(("Fc", "Qax", "Qlat")):
            ax.scatter(
                math.sqrt(emp_var[i]), math.sqrt(pred["variances"][i]),
                color=couleurs[geom], marker=marqueurs[nom], s=52,
                edgecolor="white", linewidth=0.45, zorder=3,
            )
    bornes = []
    bornes.extend(np.sqrt(emp_var).tolist())
    for pred in predictions.values():
        bornes.extend(np.sqrt(pred["variances"]).tolist())
    mini = min(bornes) * 0.72
    maxi = max(bornes) * 1.38
    ax.plot([mini, maxi], [mini, maxi], color="black", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mini, maxi)
    ax.set_ylim(mini, maxi)
    ax.set_xlabel("empirical standard deviation (400 reconstructions)")
    ax.set_ylabel("predicted standard deviation")
    ax.set_title(r"(iii) $\Gamma$ propagated to the standard deviations")
    leg_geom = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=couleurs[n],
               markeredgecolor="none", label=n, markersize=7)
        for n in couleurs
    ]
    leg_obs = [
        Line2D([0], [0], marker=marqueurs[n], color="#333333", linestyle="none",
               label=n, markersize=6) for n in marqueurs
    ]
    premiere = ax.legend(handles=leg_geom, loc="upper left", fontsize=7, ncol=2)
    ax.add_artist(premiere)
    ax.legend(handles=leg_obs, loc="lower right", fontsize=7)
    ax.grid(True, which="both", alpha=0.2)

    ax = axes[1, 1]
    emp_shift = valeurs.mean(axis=0) - prep["base_means"]
    pred_shift = predictions["K_cal"]["mean_shifts"]
    positions = np.arange(2)
    pred_indices = [1, 2]
    pred_q = np.quantile(pred_mean_boot[:, pred_indices], [0.025, 0.975], axis=0)
    emp_q = np.quantile(emp_mean_boot[:, pred_indices], [0.025, 0.975], axis=0)
    pred_y = pred_shift[pred_indices]
    emp_y = emp_shift[pred_indices]
    pred_err = np.vstack((pred_y - pred_q[0], pred_q[1] - pred_y))
    emp_err = np.vstack((emp_y - emp_q[0], emp_q[1] - emp_y))
    ax.errorbar(
        positions - 0.10, pred_y, yerr=pred_err, fmt="o", color="#0077b6",
        capsize=4, label="predicted by K_cal (calibration bootstrap)",
    )
    ax.errorbar(
        positions + 0.10, emp_y, yerr=emp_err, fmt="s", color="#d00000",
        capsize=4, label="empirical (test bootstrap)",
    )
    ax.set_xticks(positions, ["Q_ax", "Q_lat"])
    ax.set_ylabel("mean shift")
    ax.set_title(
        r"(iv) $A_{\mathrm{err}}$ propagated, and the error on the error operator"
    )
    ax.axhline(0.0, color="black", lw=0.7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "BrainWeb-MS: held-out calibration and exact propagation of "
        r"$(A_{\mathrm{err}}, \Gamma)$",
        fontsize=13,
    )
    fig.savefig(HERE / "exp2_figure.pdf")
    fig.savefig(HERE / "exp2_figure.png", dpi=220)
    plt.close(fig)
    return float(time.perf_counter() - debut)


def jsonise(obj):
    if isinstance(obj, dict):
        return {str(k): jsonise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def ecrit_json(resultats: dict) -> None:
    (HERE / "exp2_results.json").write_text(
        json.dumps(jsonise(resultats), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    debut_total = time.perf_counter()
    date_debut = datetime.now().astimezone().isoformat(timespec="seconds")
    print("Chargement, masque fixe et test dense 8x8...", flush=True)
    normal_brut = charge_volume("phantom_normal_crisp", "<i2")
    if np.any(normal_brut % 455):
        raise ValueError("Le volume normal n'est pas codé en classes x455")
    vol_normal = (normal_brut // 455).astype(np.uint8)
    vol_ms = charge_volume("phantom_msles2_crisp", np.uint8)
    masque = masque_lignes(FRAC_LINES, CENTER_LINES, np.random.default_rng(SEEDS["mask"]))
    dense = test_dense_interne()
    if not dense["pass"]:
        raise AssertionError("Une identité dense obligatoire a échoué")

    print("Calibration centrée et porte C0...", flush=True)
    cal = calibre_et_prepare_c0(vol_normal, masque)
    gain, _ = gain_posterieur(masque)
    ktrue = gain**2 * SIGMA_NOISE**2
    comparaison = comparaison_ktrue(cal["khat"], ktrue, masque)
    geoms, diag_geoms = construit_geometries(cal["khat"])
    moyenne_khat_observee = float(np.mean(cal["khat"][masque]))
    delta2 = float((SPECTRAL_FLOOR_FACTOR * moyenne_khat_observee) ** 2)

    sequence = np.random.SeedSequence(SEEDS["bootstrap"])
    enfants = sequence.spawn(4)
    rng_c0, rng_cal_boot, rng_test_boot, rng_c3 = [
        np.random.default_rng(s) for s in enfants
    ]
    c0 = evalue_c0(cal["c0_centered_hat"], geoms, delta2, masque, rng_c0)
    print(
        f"C0={'PASS' if c0['pass'] else 'FAIL'} ; "
        f"Delta NLL/champ={c0['mean_difference_per_field']:.6g}", flush=True,
    )

    sampler = distances_loi_sampler(vol_ms)
    base = {
        "schema_version": 1,
        "experiment": "BrainWeb experience 2: calibrated (A,Gamma) propagation",
        "run": {
            "started_at": date_debut,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "data": {
            "normal": empreinte(BRAINWEB_DIR / "phantom_normal_crisp.rawb.gz"),
            "ms": empreinte(BRAINWEB_DIR / "phantom_msles2_crisp.rawb.gz"),
            "volume_shape_zyx": list(SHAPE),
            "normal_classes_after_division_by_455": [int(x) for x in np.unique(vol_normal)],
            "ms_classes": [int(x) for x in np.unique(vol_ms)],
            "ms_volume_total_lesion_voxels_label_10": int(np.sum(vol_ms == 10)),
        },
        "seeds": {
            **SEEDS,
            "bootstrap_substreams": {
                "construction": "numpy SeedSequence(271828).spawn(4)",
                "order": ["C0", "calibration", "test", "C3"],
                "spawn_keys": [list(s.spawn_key) for s in enfants],
            },
            "sampler_inherited_mask": 7,
            "sampler_inherited_observation": 8,
        },
        "parameters": {
            "calibration_slices": list(Z_CAL),
            "held_out_C0_slices": list(Z_C0),
            "calibration_noise_repetitions_per_slice": J_CAL,
            "test_slice": Z_TEST,
            "test_noise_repetitions": J_TEST,
            "bootstrap_repetitions": B_BOOT,
            "fraction_noncentral_line_pairs": FRAC_LINES,
            "fully_sampled_center_half_width": CENTER_LINES,
            "fixed_sampled_line_count": int(masque[:, 0].sum()),
            "observation_sigma": SIGMA_NOISE,
            "prior_lambda": PRIOR_LAMBDA,
            "prior_epsilon": PRIOR_EPS,
            "prior_mean": PRIOR_MEAN,
            "radial_annuli": RADIAL_BINS,
            "spectral_floor_formula": "delta2 = (1e-3 * mean(khat on observed modes))^2",
            "spectral_floor_delta2": delta2,
        },
        "validation": {
            "dense_8x8": dense,
            "fixed_mask": {
                "same_array_used_calibration_C0_and_test": True,
                "hermitian": bool(np.array_equal(masque, masque[np.ix_((-np.arange(NY)) % NY, (-np.arange(NX)) % NX)])),
                "sampled_lines": int(masque[:, 0].sum()),
                "mask_sha256_uint8": hashlib.sha256(np.ascontiguousarray(masque, dtype=np.uint8).tobytes()).hexdigest(),
            },
            "calibration_centering": {
                "J_over_J_minus_1_correction": cal["centering_correction"],
                "sum_s_J_minus_1_denominator": cal["centering_denominator"],
                "max_abs_khat_between_equivalent_correction_routes": cal["max_abs_khat_two_correction_routes"],
                "max_abs_fft_spatial_vs_direct_centered": cal["max_abs_fft_spatial_vs_direct_centered"],
                "max_abs_centered_coefficient_unobserved": cal["max_abs_centered_coefficient_unobserved"],
                "C0_centered_fields_sqrt_J_over_J_minus_1_scale": cal["c0_centered_field_marginal_scale"],
                "analytic_unobserved_value": 0.0,
            },
        },
        "calibration": {
            "interpretation": (
                "b_hat_s is the slice/anatomy-dependent A-type bias; K_hat is "
                "estimated only from within-slice centered random residuals"
            ),
            "bias_by_calibration_slice": cal["bias_summaries"],
            "bias_by_held_out_slice_for_centering_only": cal["c0_bias_summaries"],
            "k_hat": {
                "shape": [NY, NX],
                "fft_convention": "numpy fft2 norm=ortho, row-major ky,kx",
                "estimator": "sum |fft(r_sj-b_hat_s)|^2 / (5*(24-1))",
                "trace_sum_spectrum": float(np.sum(cal["khat"])),
                "mean_observed": moyenne_khat_observee,
                "mean_all_modes": float(np.mean(cal["khat"])),
                "sha256_float64": empreinte_tableau(cal["khat"]),
                "values": cal["khat"],
            },
            "k_true": {
                "formula": "mask * sigma_n^2 / (1 + sigma_n^2*(lambda*LAP^2+epsilon))^2",
                "trace_sum_spectrum": float(np.sum(ktrue)),
                "mean_observed": float(np.mean(ktrue[masque])),
                "max_unobserved": float(np.max(ktrue[~masque])),
                "sha256_float64": empreinte_tableau(ktrue),
            },
            "k_hat_vs_k_true": comparaison,
            "spectral_likelihood_floor_delta2": delta2,
            "duration_seconds": cal["duration_seconds"],
        },
        "geometries": diag_geoms,
        "criteria": {"C0": c0},
        "sampler_law_distances": sampler,
    }

    if not c0["pass"]:
        base["criteria"].update({
            "C1": {"pass": False, "not_run": "C0 failed"},
            "C2": {"pass": False, "not_run": "C0 failed"},
            "C3": {"pass": False, "not_run": "C0 failed"},
            "C4": {"pass": False, "not_run": "C0 failed; lesion test forbidden"},
        })
        base["verdict"] = {
            "go": False,
            "label": "NO-GO de puissance",
            "reason": "C0 failed; lesion test not launched",
        }
        base["run"]["duration_seconds"] = float(time.perf_counter() - debut_total)
        base["durations_seconds"] = {
            "dense_internal": dense["duration_seconds"],
            "calibration_and_C0_data": cal["duration_seconds"],
            "C0": c0["duration_seconds"],
            "sampler_law_distances": sampler["duration_seconds"],
            "total_before_json_write": base["run"]["duration_seconds"],
        }
        base["outputs"] = ["exp2_results.json"]
        ecrit_json(base)
        print("STOP pré-spécifié aprè échec C0.", flush=True)
        return

    print("Test lésionnel, moments fermés et doubles bootstraps...", flush=True)
    xstar = coupe(vol_ms, Z_TEST)
    m, _ = moyenne_sans_bruit(xstar, masque)
    obs = construit_observables(vol_ms)
    prep = prepare_moments(m, obs)
    valeurs, diag_test = simule_test(xstar, masque, m, obs)

    pred_geoms = {nom: prediction_moments(k, prep) for nom, k in geoms.items()}
    pred_geoms["K_true"] = prediction_moments(ktrue, prep)
    pred_mean_boot, pred_var_boot, duree_cal_boot = bootstrap_calibration_predictions(
        cal["powers"], prep, rng_cal_boot
    )
    emp_mean = valeurs.mean(axis=0) - prep["base_means"]
    emp_var = valeurs.var(axis=0, ddof=1)
    emp_mean_boot, emp_var_boot, duree_test_boot = bootstrap_test_observables(
        valeurs, prep["base_means"], rng_test_boot
    )
    pred_cal = pred_geoms["K_cal"]
    c1, c2, simultane = criteres_c1_c2(
        pred_cal["mean_shifts"], pred_cal["variances"], emp_mean, emp_var,
        pred_mean_boot, pred_var_boot, emp_mean_boot, emp_var_boot,
    )
    geoms_oracle = {**geoms, "K_true": ktrue}
    c3, scores = evalue_c3(valeurs, geoms_oracle, delta2, prep, rng_c3)

    print("Génération de la figure quatre panneaux...", flush=True)
    duree_figure = fait_figure(
        xstar, obs, cal["khat"], geoms["K_radial"], delta2, valeurs,
        pred_geoms, pred_mean_boot, emp_mean_boot, prep,
    )
    c4_passe = all(
        (HERE / nom).is_file() and (HERE / nom).stat().st_size > 0
        for nom in ("exp2_figure.pdf", "exp2_figure.png")
    )
    c4 = {
        "name": "C4_four_panel_figure",
        "pass": bool(c4_passe),
        "generated_files": ["exp2_figure.pdf", "exp2_figure.png"],
        "panels": 4,
        "png_native_shape_height_width_channels": [
            int(x) for x in plt.imread(HERE / "exp2_figure.png").shape
        ],
        "files": {
            nom: {
                "bytes": int((HERE / nom).stat().st_size),
                "sha256": hashlib.sha256((HERE / nom).read_bytes()).hexdigest(),
            }
            for nom in ("exp2_figure.pdf", "exp2_figure.png")
        },
        "manual_native_inspection": "required post-run and recorded in RAPPORT.md",
    }

    noms_obs = ["F_c", "Q_ax", "Q_lat"]
    table_geoms = {}
    for nom, pred in pred_geoms.items():
        table_geoms[nom] = {
            "trace_sum_spectrum": float(np.sum(geoms_oracle[nom])),
            "trace_relative_to_Khat": float(np.sum(geoms_oracle[nom]) / np.sum(cal["khat"])),
            "mean_shifts": {noms_obs[i]: float(pred["mean_shifts"][i]) for i in range(3)},
            "variances": {noms_obs[i]: float(pred["variances"][i]) for i in range(3)},
            "standard_deviations": {
                noms_obs[i]: float(math.sqrt(pred["variances"][i])) for i in range(3)
            },
            "covariance_matrix": [[float(x) for x in ligne] for ligne in pred["covariance"]],
            "closed_formula_components": pred["components"],
            "law_score": scores[nom],
        }

    base["validation"]["test_reconstruction_linearity"] = diag_test
    base["data"].update({
        "test_slice": Z_TEST,
        "test_slice_lesion_pixels": int(obs["lesions"].sum()),
        "test_roi_pixels": int(obs["roi"].sum()),
        "test_wm_ring_pixels": int(obs["wm_ring"].sum()),
    })
    base["test"] = {
        "source": "MS phantom slice 101, same fixed mask, 400 independent noises",
        "observable_order": noms_obs,
        "baseline_at_m": {noms_obs[i]: float(prep["base_means"][i]) for i in range(3)},
        "empirical_mean_shifts": {noms_obs[i]: float(emp_mean[i]) for i in range(3)},
        "empirical_variances": {noms_obs[i]: float(emp_var[i]) for i in range(3)},
        "empirical_standard_deviations": {
            noms_obs[i]: float(math.sqrt(emp_var[i])) for i in range(3)
        },
        "empirical_observable_values_sha256_float64": empreinte_tableau(valeurs),
        "calibration_bootstrap_prediction_mean_shifts": resume_bootstrap(
            noms_obs, pred_cal["mean_shifts"], pred_mean_boot
        ),
        "calibration_bootstrap_prediction_variances": resume_bootstrap(
            noms_obs, pred_cal["variances"], pred_var_boot
        ),
        "test_bootstrap_empirical_mean_shifts": resume_bootstrap(
            noms_obs, emp_mean, emp_mean_boot
        ),
        "test_bootstrap_empirical_variances": resume_bootstrap(
            noms_obs, emp_var, emp_var_boot
        ),
        "simultaneous_bootstrap_band": simultane,
        "geometry_table_4_plus_oracle": table_geoms,
        "law_scores_C3": scores,
    }
    base["criteria"].update({"C1": c1, "C2": c2, "C3": c3, "C4": c4})
    go = bool(all(base["criteria"][nom]["pass"] for nom in ("C0", "C1", "C2", "C3", "C4")))
    base["verdict"] = {
        "go": go,
        "label": "GO" if go else "NO-GO",
        "reason": "all pre-specified C0-C4 pass" if go else "at least one pre-specified criterion fails",
    }
    duree_total = float(time.perf_counter() - debut_total)
    base["run"]["duration_seconds"] = duree_total
    base["durations_seconds"] = {
        "dense_internal": dense["duration_seconds"],
        "calibration_and_C0_data": cal["duration_seconds"],
        "C0": c0["duration_seconds"],
        "lesion_test_simulation": diag_test["duration_seconds"],
        "calibration_bootstrap_propagation": duree_cal_boot,
        "test_bootstrap": duree_test_boot,
        "C3": c3["duration_seconds"],
        "sampler_law_distances": sampler["duration_seconds"],
        "figure": duree_figure,
        "total_before_json_write": duree_total,
    }
    base["outputs"] = [
        "exp2_results.json", "exp2_figure.pdf", "exp2_figure.png"
    ]
    ecrit_json(base)
    print(
        "Critères: " + ", ".join(
            f"{nom}={'PASS' if base['criteria'][nom]['pass'] else 'FAIL'}"
            for nom in ("C0", "C1", "C2", "C3", "C4")
        ), flush=True,
    )
    print(f"Verdict final: {base['verdict']['label']} ; durée={duree_total:.2f} s", flush=True)


if __name__ == "__main__":
    main()
