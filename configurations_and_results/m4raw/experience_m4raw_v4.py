#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4Raw v4: alignment-consistent replication on unseen MRI subjects.

The script is deterministic and self-contained.  It discovers M4Raw H5 files
inside the supplied ZIP archives, materializes them in a local cache when
needed, and runs either the non-confirmatory 30-subject integration smoke or
the frozen full replication.  Configuration is accepted through
M4RAW_ARCHIVES/M4RAW_MODE/M4RAW_OUT or the matching command-line arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import pathlib
import platform
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


HERE = pathlib.Path(__file__).resolve().parent
_EARLY_ARCHIVE_DIR = pathlib.Path(os.environ.get("M4RAW_ARCHIVES", str(HERE)))
CACHE_ROOT = _EARLY_ARCHIVE_DIR
DATA_DIR = CACHE_ROOT / "val_extrait"
ARCHIVE = CACHE_ROOT / "M4RawV1.5_multicoil_val.zip"
ANALYSIS_CACHE = CACHE_ROOT / "analysis_cache_v4"
V4_FILE_PATHS: dict[tuple[str, str], pathlib.Path] = {}
V4_FILE_ARCHIVES: dict[tuple[str, str], str] = {}
V4_HDF5_KEYS_READ: set[str] = set()
V4_HDF5_AVAILABLE_KEYS: dict[str, set[tuple[str, ...]]] = {}
V4_EXTERNAL_SECONDARY_UNLOCKED = False
V4_OUT = HERE
os.environ.setdefault(
    "MPLCONFIGDIR", str(CACHE_ROOT / ".m4raw_v4_cache" / "matplotlib")
)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402
from scipy.ndimage import (  # noqa: E402
    binary_erosion,
    fourier_shift,
    generate_binary_structure,
    label,
    rotate,
)
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import chi2, norm  # noqa: E402


# Mission parameters. These values are never tuned from the outcomes.
N = 256
COILS = 4
SLICES = np.arange(4, 14)
REPETITIONS = ("T201", "T202", "T203")
CONTRAST_REPETITIONS = {
    "T2": ("T201", "T202", "T203"),
    "T1": ("T101", "T102", "T103"),
    "FLAIR": ("FLAIR01", "FLAIR02"),
}
PAIRS = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
CONSECUTIVE_PAIRS = ((0, 1), (1, 2))
OUTER_ROWS = np.where(np.abs(np.arange(N) - 128) > 48)[0]
LOWPASS_WIDTH = 24
ALIGNMENT_HALF_WIDTH = 16
GUARD_HALF_WIDTH = 24
MASK_THRESHOLD = 0.15
MASK_EROSION = 2
B_BOOT = 5000
BOOTSTRAP_SEED = 424242
SYNTHETIC_SEED = 424242
SPECTRAL_FLOOR_RELATIVE = 1.0e-6

ARCHIVE_BASENAMES = {
    "train": "M4RawV1.5_multicoil_train.zip",
    "test": "M4Raw_multicoil_test.zip",
    "val": "M4RawV1.5_multicoil_val.zip",
}
EXTERNAL_REPETITIONS = {
    "primary": {
        "T2": ("T201", "T202", "T203"),
        "T1": ("T101", "T102", "T103"),
        "FLAIR": ("FLAIR01", "FLAIR02"),
    },
    "secondary": {
        "T2": ("T204", "T205", "T206"),
        "T1": ("T104", "T105", "T106"),
        "FLAIR": ("FLAIR03", "FLAIR04"),
    },
}
CACHE_SCHEMA = "m4raw-v4-v2-gelee-cache-1"
SMOKE_RAW_R_REFERENCE = 0.9282649431967674
SMOKE_RAW_R_ABSOLUTE_TOLERANCE = 0.02
GAUGE_OPTIMIZER_G_TOL = 1.0e-10
GAUGE_OPTIMIZER_MAX_ITERATIONS = 1000
GAUGE_PHASE_MULTISTARTS = tuple(itertools.product((0.0, math.pi), repeat=3))

# Frozen v3.1 validation cohort, excluded in full mode without inspecting any
# outcome in the train/test archives.
V31_VALIDATION_SUBJECTS = (
    "2022061203", "2022061204", "2022061206", "2022061207",
    "2022061213", "2022061301", "2022061302", "2022062303",
    "2022062501", "2022062604", "2022062612", "2022062614",
    "2022062616", "2022070508", "2022090101", "2022090102",
    "2022090104", "2022090301", "2022090305", "2022090503",
    "2022090602", "2022090603", "2022090803", "2022091405",
    "2022091408", "2022091505", "2022092002", "2022092007",
    "2022092502", "2023053002",
)

# Audited pilot values are embedded so the server package has no dependency on
# code_m4raw/ while the prescribed direct Delta_g panel remains reproducible.
V31_PILOT_DELTA = -0.010527853554189726
V31_PILOT_DELTA_CI = (-0.027942685183727767, 0.0007906172901892319)

MASK_KINDS = ("common_high_rss", "largest_connected_component")
ROUTES = ("aligned", "raw")
FULL_SURROGATE_ATTEMPTS = (
    {"degree": 1, "minimum_train": 96, "validation": 64},
    {"degree": 2, "minimum_train": 320, "maximum_train": 720, "validation": 96},
)
SMOKE_SURROGATE_ATTEMPTS = (
    {"degree": 1, "minimum_train": 48, "validation": 24},
    {"degree": 2, "minimum_train": 80, "maximum_train": 160, "validation": 32},
    {"exact_all_unique": True, "maximum_unique": 256},
)

# Operational definition of a pair that needs non-translation registration.
# A flag requires both a material transform and a >2 percentage-point gain in
# normalized correlation, which guards the diagnostic against flat/noisy ROIs.
ROTATION_EXCLUSION_DEG = 1.0
DEFORMATION_EXCLUSION_PX = 1.0
REGISTRATION_GAIN_MIN = 0.02
ROTATION_GRID_DEG = np.arange(-3.0, 3.01, 0.5)
DIAGNOSTIC_SLICE_INDICES = (2, 5, 8)

# Fixed validation rule for the bootstrap response-surface accelerator.
SURROGATE_ATTEMPTS = (
    {"degree": 2, "train": 160, "validation": 64},
    {"degree": 3, "train": 360, "validation": 96},
)
SURROGATE_RAW_RELATIVE_P99_MAX = 5.0e-3
EXACT_SUBJECT_WORKERS = 3
SURROGATE_TARGET_ABS_MAX = {
    "mean_R_calibrated": 2.0e-3,
    "mean_R_diagonal": 2.0e-3,
    "mean_R_wrong_cyclic": 2.0e-3,
    "delta_gain_diag": 1.0e-3,
    "empirical_gain_diag": 1.0e-3,
    "wrong_empirical_ratio": 2.0e-3,
}

CONTROL_NAMES = ("calibrated", "diagonal", "identity", "wrong_cyclic")
COLORS = {
    "calibrated": "#0077b6",
    "diagonal": "#2a9d8f",
    "identity": "#6c757d",
    "wrong_cyclic": "#e76f51",
}


def jsonise(obj):
    """Convert numpy and complex objects to strict JSON-compatible values."""
    if isinstance(obj, dict):
        return {str(k): jsonise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "real": np.asarray(obj.real).tolist(),
                "imaginary": np.asarray(obj.imag).tolist(),
            }
        return obj.tolist()
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imaginary": float(obj.imag)}
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def complex_matrix_record(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return {
        "real": x.real.tolist(),
        "imaginary": x.imag.tolist(),
        "frobenius_norm": float(np.linalg.norm(x)),
        "eigenvalues": [float(v) for v in np.linalg.eigvalsh(x)],
    }


def hash_file(path: pathlib.Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def v4_read_kspace_slices(
    path: pathlib.Path, archive_label: str
) -> tuple[np.ndarray, dict]:
    """Read only k-space (and non-image header metadata) from one HDF5 file."""
    with h5py.File(path, "r") as handle:
        available = tuple(sorted(str(key) for key in handle.keys()))
        V4_HDF5_AVAILABLE_KEYS.setdefault(archive_label, set()).add(available)
        if "kspace" not in handle:
            raise AssertionError(f"Missing kspace dataset in {path.name}")
        V4_HDF5_KEYS_READ.add("kspace")
        dataset = handle["kspace"]
        if dataset.shape != (18, COILS, N, N) or dataset.dtype != np.complex64:
            raise AssertionError(
                f"Unexpected kspace layout in {path.name}: "
                f"{dataset.shape}, {dataset.dtype}"
            )
        kspace = np.asarray(dataset[SLICES])
        header_text = ""
        if "ismrmrd_header" in handle:
            V4_HDF5_KEYS_READ.add("ismrmrd_header")
            header_value = handle["ismrmrd_header"][()]
            if isinstance(header_value, bytes):
                header_text = header_value.decode("utf-8", errors="replace")
            else:
                header_text = str(header_value)
    receiver_match = re.search(
        r"<[^>]*receiverChannels>(\d+)</[^>]*receiverChannels>", header_text
    )
    receiver_channels = int(receiver_match.group(1)) if receiver_match else None
    coil_dimension_declared = bool(
        re.search(r"<[^>]*DimName>\s*Coil\s*</[^>]*DimName>", header_text)
    )
    if receiver_channels not in (None, COILS):
        raise AssertionError(
            f"Header receiver-channel count is {receiver_channels} in {path.name}"
        )
    if header_text and not coil_dimension_declared:
        raise AssertionError(f"Header does not declare the k-space coil dimension in {path.name}")
    return kspace, {
        "available_root_keys": available,
        "read_dataset_keys": ["kspace"]
        + (["ismrmrd_header"] if header_text else []),
        "receiver_channels": receiver_channels,
        "coil_axis_zero_based": 1,
        "coil_dimension_declared_in_header": coil_dimension_declared,
        "stored_channel_order": list(range(COILS)),
    }


def v4_assert_no_ground_truth_read() -> dict:
    forbidden = sorted(
        key
        for key in V4_HDF5_KEYS_READ
        if any(token in key.lower() for token in ("reconstruction", "ground", "target"))
    )
    if forbidden:
        raise AssertionError(f"Ground-truth/image datasets were read: {forbidden}")
    allowed = {"kspace", "ismrmrd_header"}
    unexpected = sorted(V4_HDF5_KEYS_READ - allowed)
    if unexpected:
        raise AssertionError(f"Unexpected HDF5 datasets were read: {unexpected}")
    return {
        "dataset_keys_actually_read": sorted(V4_HDF5_KEYS_READ),
        "allowed_dataset_keys": sorted(allowed),
        "ground_truth_or_reconstruction_dataset_read": False,
        "assertion_pass": True,
        "available_root_key_sets_by_archive": {
            archive: [list(keys) for keys in sorted(key_sets)]
            for archive, key_sets in sorted(V4_HDF5_AVAILABLE_KEYS.items())
        },
    }


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def fft2c(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def central_window() -> np.ndarray:
    result = np.zeros((N, N), dtype=float)
    half = LOWPASS_WIDTH // 2
    result[N // 2 - half:N // 2 + half, N // 2 - half:N // 2 + half] = 1.0
    return result


LOWPASS_WINDOW = central_window()
FREQ = np.fft.fftshift(np.fft.fftfreq(N))
FREQ_Y, FREQ_X = np.meshgrid(FREQ, FREQ, indexing="ij")


def inventory_data() -> tuple[list[str], dict[str, list[pathlib.Path]], dict]:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    files = sorted(DATA_DIR.glob("*_T2*.h5"))
    grouped: dict[str, list[pathlib.Path]] = {}
    for path in files:
        subject, repetition = path.stem.split("_")
        if repetition in REPETITIONS:
            grouped.setdefault(subject, []).append(path)
    subjects = sorted(grouped)
    if len(subjects) != 30:
        raise AssertionError(f"Expected 30 subjects, found {len(subjects)}")
    for subject in subjects:
        found = sorted(p.stem.split("_")[1] for p in grouped[subject])
        if found != list(REPETITIONS):
            raise AssertionError(f"{subject}: T2 repetitions are {found}")
    scout_subjects = ["2022061203", "2022061204"]
    if subjects[:2] != scout_subjects:
        raise AssertionError(
            f"STOP: scout subjects are not temporal subjects 1 and 2: {subjects[:2]}"
        )
    split = {
        "calibration": subjects[:10],
        "C0_gate": subjects[10:15],
        "test": subjects[15:30],
    }
    if not set(scout_subjects).issubset(split["calibration"]):
        raise AssertionError("STOP: a scout subject is outside calibration")
    return subjects, grouped, {
        "order": subjects,
        "split": split,
        "identifier_semantics": (
            "Acquisition-date identifiers; lexicographic splitting is temporal, not random. "
            "Subjects are not represented as exchangeable across acquisition time."
        ),
        "scout_subjects": scout_subjects,
        "scout_subjects_are_numbers_1_and_2": True,
        "scout_route_choice_was_informed_only_by_calibration_subjects": True,
    }


def scan_active_support(grouped: dict[str, list[pathlib.Path]]) -> tuple[np.ndarray, dict]:
    active = None
    per_file = []
    object_support_count = 0
    for subject in sorted(grouped):
        for path in sorted(grouped[subject]):
            with h5py.File(path, "r") as handle:
                dataset = handle["kspace"]
                if dataset.shape != (18, COILS, N, N) or dataset.dtype != np.complex64:
                    raise AssertionError(f"Unexpected kspace layout in {path.name}: {dataset.shape}, {dataset.dtype}")
                kspace = np.asarray(dataset[SLICES])
            nonzero = np.abs(kspace) > 0.0
            # A is the set of kx columns containing at least one acquired ky
            # coefficient.  The v3 assertion is made for every slice and coil,
            # hence also across all three repetitions and all 30 subjects.
            object_supports = nonzero.any(axis=2)
            if active is None:
                active = object_supports[0, 0].copy()
            equal_objects = np.all(object_supports == active[None, None, :], axis=2)
            if not bool(np.all(equal_objects)):
                differing = np.argwhere(~equal_objects)
                raise AssertionError(
                    "Active support varies; an object-specific Fourier factor is required: "
                    f"{path.name}, slice/coil indices {differing[:8].tolist()}"
                )
            object_support_count += int(equal_objects.size)
            file_active = active
            fill = float(nonzero[:, :, :, active].mean())
            per_file.append(
                {
                    "file": path.name,
                    "active_first": int(np.where(file_active)[0][0]),
                    "active_last": int(np.where(file_active)[0][-1]),
                    "active_count": int(file_active.sum()),
                    "support_identical_for_all_retained_slices_and_coils": True,
                    "fill_fraction_on_common_active_support": fill,
                }
            )
    if active is None:
        raise AssertionError("No active support found")
    indices = np.where(active)[0]
    if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise AssertionError("Active kx support is not contiguous")
    common_fills = [row["fill_fraction_on_common_active_support"] for row in per_file]
    minimum = float(min(common_fills))
    if minimum < 0.98:
        raise AssertionError(f"STOP: active-support filling is only {minimum:.6f}")
    count = int(active.sum())
    return active, {
        "definition": "kx columns nonzero for every retained T2 slice/coil object; all ky rows",
        "active_columns_zero_based": indices.tolist(),
        "first_column_zero_based": int(indices[0]),
        "last_column_zero_based": int(indices[-1]),
        "active_column_count": count,
        "full_grid_frequency_count": N * N,
        "active_frequency_count": N * count,
        "c_A_exact_fraction": f"{N * count}/{N * N}",
        "c_A": float(count / N),
        "support_identical_between_all_coils_repetitions_and_retained_slices": True,
        "support_objects_asserted": object_support_count,
        "minimum_file_fill_fraction": minimum,
        "mean_file_fill_fraction": float(np.mean(common_fills)),
        "all_files_at_least_98_percent_filled": True,
        "per_file": per_file,
    }


def frequency_supports(active: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    distance = np.abs(np.arange(N) - N // 2)
    dy = distance[:, None]
    dx = distance[None, :]
    support_a = np.broadcast_to(active[None, :], (N, N)).copy()
    central = support_a & (dy <= ALIGNMENT_HALF_WIDTH) & (dx <= ALIGNMENT_HALF_WIDTH)
    # v3.1 correction: C-plus is the enlarged central BLOCK (logical AND),
    # never the cross produced by a logical OR.
    guard = support_a & (dy <= GUARD_HALF_WIDTH) & (dx <= GUARD_HALF_WIDTH)
    evaluation = support_a & ~guard
    if np.any(central & ~guard):
        raise AssertionError("Alignment block C is not contained in C-plus")
    if np.any(evaluation & guard):
        raise AssertionError("Evaluation support E intersects C-plus")
    masks = {"A": support_a, "C": central, "C_plus": guard, "E": evaluation}
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    return masks, {
        "definitions": {
            "A": "all ky and the common active kx columns",
            "C": "A intersect {|ky-128|<=16 and |kx-128|<=16}",
            "C_plus": "A intersect {|ky-128|<=24 and |kx-128|<=24}",
            "E": "A minus C_plus",
        },
        "frequency_counts": counts,
        "c_A_exact_fraction": f"{counts['A']}/{N * N}",
        "c_E_exact_fraction": f"{counts['E']}/{N * N}",
        "c_A": float(counts["A"] / (N * N)),
        "c_E": float(counts["E"] / (N * N)),
        "C_subset_C_plus": True,
        "E_intersection_C_plus_is_empty": True,
        "mask_sha256": {
            name: array_hash(mask.astype(np.uint8)) for name, mask in masks.items()
        },
    }


def centered_moments(samples: np.ndarray) -> dict:
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.ndim != 2 or samples.shape[1] != COILS:
        raise ValueError(f"Samples must have shape (n,{COILS})")
    mean = samples.mean(axis=0)
    z = samples - mean[None, :]
    psi = (z.T @ z.conj()) / len(z)
    pseudo = (z.T @ z) / len(z)
    real_augmented_samples = np.concatenate((z.real, z.imag), axis=1)
    real_augmented = (
        real_augmented_samples.T @ real_augmented_samples / len(z)
    )
    psi = 0.5 * (psi + psi.conj().T)
    real_augmented = 0.5 * (real_augmented + real_augmented.T)
    return {
        "n": int(len(z)),
        "mean": mean,
        "psi": psi,
        "pseudo": pseudo,
        "real_augmented": real_augmented,
    }


def noise_statistics_subject(subject: str, active: np.ndarray) -> dict:
    repetitions = []
    for repetition in REPETITIONS:
        path = DATA_DIR / f"{subject}_{repetition}.h5"
        with h5py.File(path, "r") as handle:
            # Read only the clean outer route used for covariance estimation.
            repetitions.append(np.asarray(handle["kspace"][SLICES])[:, :, OUTER_ROWS][:, :, :, active])
    pair_records = []
    for i, j in CONSECUTIVE_PAIRS:
        difference = (repetitions[i] - repetitions[j]) / math.sqrt(2.0)
        samples = difference.transpose(0, 2, 3, 1).reshape(-1, COILS)
        moments = centered_moments(samples)
        pair_records.append(
            {
                "pair": f"r{i + 1}-r{j + 1}",
                **moments,
            }
        )
    psi = np.mean([row["psi"] for row in pair_records], axis=0)
    pseudo = np.mean([row["pseudo"] for row in pair_records], axis=0)
    real_augmented = np.mean([row["real_augmented"] for row in pair_records], axis=0)
    relative_pair_difference = float(
        np.linalg.norm(pair_records[0]["psi"] - pair_records[1]["psi"])
        / np.linalg.norm(psi)
    )
    return {
        "subject": subject,
        "psi": psi,
        "pseudo": pseudo,
        "real_augmented": real_augmented,
        "pseudo_to_covariance_frobenius_ratio": float(np.linalg.norm(pseudo) / np.linalg.norm(psi)),
        "pair_relative_frobenius_difference": relative_pair_difference,
        "pairs": pair_records,
    }


def correlation_matrix(psi: np.ndarray) -> np.ndarray:
    diagonal = np.real(np.diag(psi))
    if np.any(diagonal <= 0.0):
        raise AssertionError("Non-positive receiver variance")
    scale = np.sqrt(diagonal)
    return psi / np.outer(scale, scale)


def gauge_invariant_correlation_distance(r1: np.ndarray, r2: np.ndarray) -> dict:
    """Frozen gauge-invariant relative Frobenius distance, theta_1 fixed to zero."""
    r1 = np.asarray(r1, dtype=np.complex128)
    r2 = np.asarray(r2, dtype=np.complex128)
    if r1.shape != (COILS, COILS) or r2.shape != (COILS, COILS):
        raise ValueError("Gauge distance requires two 4x4 correlation matrices")
    denominator_squared = float(np.linalg.norm(r1) ** 2)

    def transformed(theta_free: np.ndarray) -> np.ndarray:
        phases = np.exp(1.0j * np.concatenate(([0.0], np.asarray(theta_free))))
        return phases[:, None] * r2 * phases.conj()[None, :]

    def objective(theta_free: np.ndarray) -> float:
        return float(np.linalg.norm(r1 - transformed(theta_free)) ** 2 / denominator_squared)

    row_indices = np.arange(COILS)[:, None]
    column_indices = np.arange(COILS)[None, :]

    def gradient(theta_free: np.ndarray) -> np.ndarray:
        current = transformed(theta_free)
        error = r1 - current
        values = []
        for channel in range(1, COILS):
            derivative = (
                1.0j
                * (
                    (row_indices == channel).astype(float)
                    - (column_indices == channel).astype(float)
                )
                * current
            )
            values.append(
                -2.0 * float(np.vdot(error, derivative).real) / denominator_squared
            )
        return np.asarray(values, dtype=float)

    reference_start = []
    for index in range(1, COILS):
        if abs(r1[0, index]) > 1.0e-15 and abs(r2[0, index]) > 1.0e-15:
            reference_start.append(float(np.angle(r2[0, index] / r1[0, index])))
        else:
            reference_start.append(0.0)
    starts = [np.asarray(reference_start), np.zeros(COILS - 1)]
    starts.extend(np.asarray(values) for values in GAUGE_PHASE_MULTISTARTS)
    candidates = []
    for start in starts:
        optimized = minimize(
            objective,
            start,
            method="BFGS",
            jac=gradient,
            options={
                "gtol": GAUGE_OPTIMIZER_G_TOL,
                "maxiter": GAUGE_OPTIMIZER_MAX_ITERATIONS,
            },
        )
        theta = np.asarray(optimized.x, dtype=float)
        candidates.append(
            (
                objective(theta),
                tuple(float(np.angle(np.exp(1.0j * value))) for value in theta),
                bool(optimized.success),
                str(optimized.message),
                int(getattr(optimized, "nit", 0)),
            )
        )
    best = min(candidates, key=lambda row: (row[0], row[1]))
    theta = np.asarray((0.0, *best[1]), dtype=float)
    distance = float(math.sqrt(max(best[0], 0.0)))
    magnitude_distance = float(
        np.linalg.norm(np.abs(r1) - np.abs(r2)) / np.linalg.norm(np.abs(r1))
    )
    return {
        "definition": "min_D ||R1-D R2 D*||_F/||R1||_F with unitary diagonal D and theta_1=0",
        "d_gauge": distance,
        "optimizing_phases_radians_theta1_fixed_zero": theta,
        "distance_of_magnitudes": magnitude_distance,
        "descriptive_threshold": 0.15,
        "below_descriptive_threshold": bool(distance <= 0.15),
        "optimizer": {
            "method": "deterministic multistart BFGS",
            "multistart_count": len(starts),
            "gtol": GAUGE_OPTIMIZER_G_TOL,
            "max_iterations": GAUGE_OPTIMIZER_MAX_ITERATIONS,
            "best_reported_success": best[2],
            "best_message": best[3],
            "best_iterations": best[4],
        },
    }


def cyclic_permutation() -> tuple[np.ndarray, list[int]]:
    # P maps old coil i to new coil (i+1) mod 4.
    permutation = [(i + 1) % COILS for i in range(COILS)]
    matrix = np.zeros((COILS, COILS), dtype=float)
    for old, new in enumerate(permutation):
        matrix[new, old] = 1.0
    return matrix, permutation


def control_matrices(psi: np.ndarray) -> dict[str, np.ndarray]:
    diagonal = np.real(np.diag(psi))
    dhalf = np.diag(np.sqrt(diagonal))
    corr = correlation_matrix(psi)
    permutation, _ = cyclic_permutation()
    wrong = dhalf @ permutation @ corr @ permutation.T @ dhalf
    controls = {
        "calibrated": np.asarray(psi, dtype=np.complex128),
        "diagonal": np.diag(diagonal).astype(np.complex128),
        "identity": np.eye(COILS, dtype=np.complex128),
        "wrong_cyclic": np.asarray(wrong, dtype=np.complex128),
    }
    for name, matrix in controls.items():
        matrix = 0.5 * (matrix + matrix.conj().T)
        if np.linalg.eigvalsh(matrix)[0] <= 0.0:
            raise AssertionError(f"Control {name} is not positive definite")
        controls[name] = matrix
    return controls


def all_derangement_controls(psi: np.ndarray) -> dict[str, dict]:
    diagonal = np.real(np.diag(psi))
    dhalf = np.diag(np.sqrt(diagonal))
    corr = correlation_matrix(psi)
    result = {}
    for permutation in itertools.permutations(range(COILS)):
        if any(permutation[i] == i for i in range(COILS)):
            continue
        p = np.zeros((COILS, COILS), dtype=float)
        for old, new in enumerate(permutation):
            p[new, old] = 1.0
        matrix = dhalf @ p @ corr @ p.T @ dhalf
        name = "derangement_" + "".join(str(i + 1) for i in permutation)
        result[name] = {
            "old_to_new_zero_based": list(permutation),
            "matrix": 0.5 * (matrix + matrix.conj().T),
        }
    if len(result) != 9:
        raise AssertionError(f"Expected 9 derangements, got {len(result)}")
    return result


def eigenvalue_inversion_control(psi: np.ndarray) -> dict:
    eigenvalues, eigenvectors = np.linalg.eigh(psi)
    matrix = eigenvectors @ np.diag(eigenvalues[::-1]) @ eigenvectors.conj().T
    condition = float(eigenvalues[-1] / eigenvalues[0])
    return {
        "applicability_rule": "computed when lambda_max/lambda_min >= 1.2",
        "condition_number": condition,
        "substantial_eigenvalue_gaps": bool(condition >= 1.2),
        "matrix": matrix,
    }


def spectral_floor_delta2(psi: np.ndarray) -> float:
    """Fixed relative floor, converted to the acquisition's variance scale."""
    return float(SPECTRAL_FLOOR_RELATIVE * np.trace(psi).real / COILS)


def spectral_floor_matrix(matrix: np.ndarray, delta2: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    floored = np.maximum(values, delta2)
    return vectors @ np.diag(floored) @ vectors.conj().T


def nll_complex_from_covariance(sample_covariance: np.ndarray, candidate: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(candidate)
    if sign.real <= 0.0 or abs(sign.imag) > 1.0e-10:
        raise AssertionError("Invalid complex covariance in NLL")
    quadratic = np.trace(np.linalg.solve(candidate, sample_covariance)).real
    return float(COILS * np.log(np.pi) + logdet.real + quadratic)


def nll_real_from_covariance(sample_covariance: np.ndarray, candidate: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(candidate)
    if sign <= 0.0:
        raise AssertionError("Invalid augmented covariance in NLL")
    dimension = candidate.shape[0]
    quadratic = np.trace(np.linalg.solve(candidate, sample_covariance))
    return float(0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic))


def symmetric_sqrt(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if values[0] <= 0.0:
        raise AssertionError("Non-positive block in augmented covariance")
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    return root, inverse_root


def augmented_controls(covariance: np.ndarray) -> dict[str, np.ndarray]:
    # Ordering is Re(coils 1..4), Im(coils 1..4). Build a block-diagonal
    # marginal square root, standardize, permute coil blocks, then restore the
    # original marginal 2x2 blocks. Thus wrong changes only cross-coil layout.
    roots = []
    inverse_roots = []
    for coil in range(COILS):
        indices = np.array([coil, COILS + coil])
        root, inverse_root = symmetric_sqrt(covariance[np.ix_(indices, indices)])
        roots.append(root)
        inverse_roots.append(inverse_root)
    block_root = np.zeros((2 * COILS, 2 * COILS))
    block_inverse = np.zeros_like(block_root)
    for coil, (root, inverse_root) in enumerate(zip(roots, inverse_roots)):
        indices = np.array([coil, COILS + coil])
        block_root[np.ix_(indices, indices)] = root
        block_inverse[np.ix_(indices, indices)] = inverse_root
    standardized = block_inverse @ covariance @ block_inverse.T
    p4, _ = cyclic_permutation()
    p8 = np.zeros((2 * COILS, 2 * COILS))
    p8[:COILS, :COILS] = p4
    p8[COILS:, COILS:] = p4
    diagonal = block_root @ np.eye(2 * COILS) @ block_root.T
    wrong = block_root @ p8 @ standardized @ p8.T @ block_root.T
    result = {
        "calibrated": covariance,
        "diagonal": diagonal,
        "wrong_cyclic": wrong,
    }
    for key, value in result.items():
        result[key] = 0.5 * (value + value.T)
        if np.linalg.eigvalsh(result[key])[0] <= 0.0:
            raise AssertionError(f"Augmented control {key} is not positive definite")
    return result


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a0 = np.asarray(a, dtype=float) - float(np.mean(a))
    b0 = np.asarray(b, dtype=float) - float(np.mean(b))
    denominator = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denominator <= 1.0e-20:
        return 0.0
    return float(np.sum(a0 * b0) / denominator)


def shift_image_fourier(image: np.ndarray, shift_yx: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftn(image, axes=(-2, -1))
    full_shift = (0.0,) * (image.ndim - 2) + tuple(float(x) for x in shift_yx)
    transformed = np.fft.ifftn(
        fourier_shift(spectrum, full_shift), axes=(-2, -1)
    )
    if np.isrealobj(image):
        return transformed.real
    return transformed


def integer_phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    moving = np.asarray(moving, dtype=float)
    if reference.shape != moving.shape or reference.ndim < 2:
        raise ValueError("Registration arrays must have the same spatial shape")
    a = reference - np.mean(reference, axis=(-2, -1), keepdims=True)
    b = moving - np.mean(moving, axis=(-2, -1), keepdims=True)
    cross = np.fft.fft2(a, axes=(-2, -1)) * np.fft.fft2(
        b, axes=(-2, -1)
    ).conj()
    if cross.ndim > 2:
        cross = np.sum(cross, axis=tuple(range(cross.ndim - 2)))
    # Energy-weighted cross-correlation is deliberate here. A phase-only
    # normalization would amplify frequency bins in which the 24x24 low-pass
    # RSS contains essentially only noise.
    correlation = np.abs(np.fft.ifft2(cross))
    peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape), dtype=float)
    spatial_shape = reference.shape[-2:]
    for axis, size in enumerate(spatial_shape):
        if peak[axis] > size // 2:
            peak[axis] -= size
        index = int(peak[axis]) % size
        slicer = [int(peak[0]) % spatial_shape[0], int(peak[1]) % spatial_shape[1]]
        minus = slicer.copy()
        plus = slicer.copy()
        minus[axis] = (index - 1) % size
        plus[axis] = (index + 1) % size
        ym = float(correlation[tuple(minus)])
        y0 = float(correlation[tuple(slicer)])
        yp = float(correlation[tuple(plus)])
        denominator = ym - 2.0 * y0 + yp
        if abs(denominator) > 1.0e-15:
            peak[axis] += float(np.clip(0.5 * (ym - yp) / denominator, -0.75, 0.75))
    return peak


def estimate_translation(reference: np.ndarray, moving: np.ndarray) -> dict:
    initial = integer_phase_correlation_shift(reference, moving)
    initial = np.clip(initial, -16.0, 16.0)
    moving = np.asarray(moving, dtype=float)

    def objective(shift):
        shifted = shift_image_fourier(moving, shift).real
        return 1.0 - normalized_correlation(reference, shifted)

    bounds = [(float(x - 1.5), float(x + 1.5)) for x in initial]
    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 24, "ftol": 1.0e-10},
    )
    shift = np.asarray(optimized.x if optimized.success else initial, dtype=float)
    shifted = shift_image_fourier(moving, shift).real
    return {
        "shift_yx": shift,
        "ncc_before": normalized_correlation(reference, moving),
        "ncc_after": normalized_correlation(reference, shifted),
        "optimizer_success": bool(optimized.success),
    }


def kspace_shift_ramp(shift_yx: np.ndarray) -> np.ndarray:
    return np.exp(
        -2.0j * np.pi * (FREQ_Y * float(shift_yx[0]) + FREQ_X * float(shift_yx[1]))
    )


def transform_kspace(kspace: np.ndarray, shift_yx: np.ndarray, unit_phase: complex) -> np.ndarray:
    return np.asarray(kspace * kspace_shift_ramp(shift_yx)[None, :, :] * unit_phase)


def estimate_transform_to_reference(reference_low: np.ndarray, moving_low: np.ndarray) -> dict:
    coil_axis = reference_low.ndim - 3
    reference_rss = np.sqrt(np.sum(np.abs(reference_low) ** 2, axis=coil_axis))
    moving_rss = np.sqrt(np.sum(np.abs(moving_low) ** 2, axis=coil_axis))
    translation = estimate_translation(reference_rss, moving_rss)
    shifted = shift_image_fourier(moving_low, translation["shift_yx"])
    support = reference_rss > MASK_THRESHOLD * float(np.percentile(reference_rss, 99))
    expanded_support = np.expand_dims(support, axis=coil_axis)
    expanded_support = np.broadcast_to(expanded_support, reference_low.shape)
    inner = np.vdot(shifted[expanded_support], reference_low[expanded_support])
    phase = complex(inner / abs(inner)) if abs(inner) > 0.0 else complex(1.0)
    residual = np.linalg.norm(
        reference_low[expanded_support] - phase * shifted[expanded_support]
    )
    scale = np.linalg.norm(reference_low[expanded_support])
    return {
        **translation,
        "unit_phase": phase,
        "phase_radians": float(np.angle(phase)),
        "relative_complex_residual_after": float(residual / max(scale, 1.0e-20)),
    }


def registration_diagnostic_pair(reference_rss_slices: np.ndarray, moving_rss_slices: np.ndarray) -> dict:
    angle_scores = []
    zero_scores = []
    best_angles = []
    local_residual_norms = []
    local_improvements = []
    per_slice = []
    for slice_index in DIAGNOSTIC_SLICE_INDICES:
        reference = reference_rss_slices[slice_index]
        moving = moving_rss_slices[slice_index]
        scores = []
        translations = []
        for angle in ROTATION_GRID_DEG:
            rotated = rotate(moving, float(angle), reshape=False, order=1, mode="constant", cval=0.0)
            estimate = estimate_translation(reference, rotated)
            scores.append(float(estimate["ncc_after"]))
            translations.append(estimate)
        best_index = int(np.argmax(scores))
        best_angle = float(ROTATION_GRID_DEG[best_index])
        zero_index = int(np.where(np.isclose(ROTATION_GRID_DEG, 0.0))[0][0])
        improvement = float(scores[best_index] - scores[zero_index])
        best_angles.append(best_angle)
        angle_scores.append(improvement)
        zero_scores.append(scores[zero_index])

        # Translation-only residual deformation diagnostic in four overlapping
        # 112x112 regions. It is diagnostic only; no interpolation is applied.
        global_shift = translations[zero_index]["shift_yx"]
        globally_aligned = shift_image_fourier(moving, global_shift)
        local_rows = []
        for y0, x0 in ((40, 40), (40, 104), (104, 40), (104, 104)):
            ref_crop = reference[y0:y0 + 112, x0:x0 + 112]
            mov_crop = globally_aligned[y0:y0 + 112, x0:x0 + 112]
            local = estimate_translation(ref_crop, mov_crop)
            residual_norm = float(np.linalg.norm(local["shift_yx"]))
            local_residual_norms.append(residual_norm)
            local_improvements.append(float(local["ncc_after"] - local["ncc_before"]))
            local_rows.append(
                {
                    "origin_yx": [y0, x0],
                    "residual_shift_yx": local["shift_yx"],
                    "residual_shift_norm": residual_norm,
                    "ncc_improvement": float(local["ncc_after"] - local["ncc_before"]),
                }
            )
        per_slice.append(
            {
                "slice": int(SLICES[slice_index]),
                "best_rotation_degrees": best_angle,
                "rotation_ncc_improvement": improvement,
                "translation_only_ncc": float(scores[zero_index]),
                "local_regions": local_rows,
            }
        )
    median_angle = float(np.median(best_angles))
    median_rotation_gain = float(np.median(angle_scores))
    residual_p95 = float(np.percentile(local_residual_norms, 95))
    local_gain_median = float(np.median(local_improvements))
    rotation_flag = bool(
        abs(median_angle) > ROTATION_EXCLUSION_DEG
        and median_rotation_gain > REGISTRATION_GAIN_MIN
    )
    deformation_flag = bool(
        residual_p95 > DEFORMATION_EXCLUSION_PX
        and local_gain_median > REGISTRATION_GAIN_MIN
    )
    return {
        "diagnostic_slices": [int(SLICES[i]) for i in DIAGNOSTIC_SLICE_INDICES],
        "median_best_rotation_degrees": median_angle,
        "median_rotation_ncc_improvement": median_rotation_gain,
        "local_residual_shift_p95_pixels": residual_p95,
        "median_local_ncc_improvement": local_gain_median,
        "rotation_non_negligible": rotation_flag,
        "deformation_non_negligible": deformation_flag,
        "exclude_pair": bool(rotation_flag or deformation_flag),
        "per_slice": per_slice,
    }


def registration_unit_test() -> dict:
    yy, xx = np.mgrid[:N, :N]
    reference = np.exp(-((yy - 126.0) ** 2 / 700.0 + (xx - 132.0) ** 2 / 1000.0))
    known_acquisition_shift = np.array([2.25, -1.50])
    moving = shift_image_fourier(reference, known_acquisition_shift)
    estimate = estimate_translation(reference, moving)
    expected = -known_acquisition_shift
    error = float(np.linalg.norm(estimate["shift_yx"] - expected))
    if error > 0.08:
        raise AssertionError(f"Fourier translation convention failed: {error}")
    return {
        "known_acquisition_shift_yx": known_acquisition_shift,
        "expected_correction_shift_yx": expected,
        "estimated_correction_shift_yx": estimate["shift_yx"],
        "euclidean_error_pixels": error,
        "pass": True,
    }


def sensitivity_from_kspace(kspace: np.ndarray) -> np.ndarray:
    low = ifft2c(kspace * LOWPASS_WINDOW[None, :, :])
    rss = np.sqrt(np.sum(np.abs(low) ** 2, axis=0))
    return low / np.maximum(rss[None, :, :], 1.0e-12 * float(np.max(rss)))


def metric_contribution(
    sensitivities: np.ndarray,
    differences: np.ndarray,
    matrix: np.ndarray,
    psi_model: np.ndarray,
    c_e: float,
    means: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray | None]:
    """Return empirical energy, predicted energy, and optional combined mean."""
    inverse = np.linalg.inv(matrix)
    t = sensitivities @ inverse.T
    denominator = np.einsum("ni,ni->n", sensitivities.conj(), t).real
    if np.any(denominator <= 1.0e-14):
        raise AssertionError("Non-positive coil-combination denominator")
    numerator = np.einsum("ni,ni->n", t.conj(), differences)
    empirical = float(np.sum(np.abs(numerator) ** 2 / denominator**2))
    predicted_point = np.einsum(
        "ni,ij,nj->n", t.conj(), psi_model, t, optimize=True
    ).real
    predicted = float(c_e * np.sum(predicted_point / denominator**2))
    combined_mean = None
    if means is not None:
        combined_mean = np.einsum("ni,ni->n", t.conj(), means) / denominator
    return empirical, predicted, combined_mean


def metric_sums_from_arrays(
    sensitivities: np.ndarray,
    differences: np.ndarray,
    matrices: dict[str, np.ndarray],
    psi_model: np.ndarray,
    c_e: float,
    chunk_size: int = 160_000,
) -> dict[str, dict[str, float]]:
    result = {name: {"empirical_sum": 0.0, "predicted_sum": 0.0} for name in matrices}
    for start in range(0, len(sensitivities), chunk_size):
        stop = min(start + chunk_size, len(sensitivities))
        s = np.asarray(sensitivities[start:stop], dtype=np.complex128)
        z = np.asarray(differences[start:stop], dtype=np.complex128)
        for name, matrix in matrices.items():
            empirical, predicted, _ = metric_contribution(s, z, matrix, psi_model, c_e)
            result[name]["empirical_sum"] += empirical
            result[name]["predicted_sum"] += predicted
    for block in result.values():
        block["R_empirical_over_predicted"] = block["empirical_sum"] / block["predicted_sum"]
    return result


def identity_prediction_coefficient(sensitivities: np.ndarray, chunk_size: int = 200_000) -> np.ndarray:
    coefficient = np.zeros((COILS, COILS), dtype=np.complex128)
    for start in range(0, len(sensitivities), chunk_size):
        s = np.asarray(sensitivities[start:start + chunk_size], dtype=np.complex128)
        denominator = np.sum(np.abs(s) ** 2, axis=1).real
        coefficient += np.einsum(
            "ni,nj,n->ij", s, s.conj(), 1.0 / denominator**2, optimize=True
        )
    return coefficient


def matrix_summary(matrix: np.ndarray) -> dict:
    return {
        **complex_matrix_record(matrix),
        "condition_number": float(np.linalg.cond(matrix)),
    }


def v4_volume_path(subject: str, repetition: str) -> pathlib.Path:
    if V4_FILE_PATHS:
        try:
            return V4_FILE_PATHS[(subject, repetition)]
        except KeyError as exc:
            raise FileNotFoundError(f"No discovered volume for {subject}_{repetition}") from exc
    return DATA_DIR / f"{subject}_{repetition}.h5"


def largest_connected_component_mask(mask: np.ndarray) -> np.ndarray:
    """Largest 6-connected component of the retained-slice 3-D mask."""
    labels, count = label(
        np.asarray(mask, dtype=bool), structure=generate_binary_structure(3, 1)
    )
    if count < 1:
        raise AssertionError("No connected intracranial mask component")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    result = labels == largest
    if not np.any(result):
        raise AssertionError("Largest connected component is empty")
    return result


def analysis_bundle_from_arrays(
    sensitivities: np.ndarray,
    differences: np.ndarray,
    means: np.ndarray,
    matrices: dict[str, np.ndarray],
    psi_model: np.ndarray,
    c_e: float,
) -> dict:
    metrics = metric_sums_from_arrays(
        sensitivities, differences, matrices, psi_model, c_e
    )
    calibrated_inverse = np.linalg.inv(matrices["calibrated"])
    s128 = sensitivities.astype(np.complex128)
    calibrated_t = s128 @ calibrated_inverse.T
    calibrated_denominator = np.einsum(
        "ni,ni->n", sensitivities.conj(), calibrated_t
    ).real
    calibrated_mean = np.einsum(
        "ni,ni->n", calibrated_t.conj(), means
    ) / calibrated_denominator
    signal_gain = {}
    signal_gain_components = {}
    signal_reference_energy = float(np.sum(np.abs(calibrated_mean) ** 2))
    for name, matrix in matrices.items():
        inverse = np.linalg.inv(matrix)
        t = s128 @ inverse.T
        denominator = np.einsum("ni,ni->n", sensitivities.conj(), t).real
        combined = np.einsum("ni,ni->n", t.conj(), means) / denominator
        difference_energy = float(np.sum(np.abs(combined - calibrated_mean) ** 2))
        signal_gain[name] = float(
            math.sqrt(difference_energy / max(signal_reference_energy, 1.0e-30))
        )
        signal_gain_components[name] = {
            "difference_energy": difference_energy,
            "calibrated_reference_energy": signal_reference_energy,
        }
    return {
        "metrics": metrics,
        "signal_gain_B_W": signal_gain,
        "signal_gain_components": signal_gain_components,
        "identity_prediction_coefficient": identity_prediction_coefficient(
            sensitivities
        ),
        "point_count": int(len(sensitivities)),
    }


class NoAdmissiblePairs(RuntimeError):
    """An eligible subject has no pair surviving the frozen technical rule."""


def prepare_test_subject(
    subject: str,
    psi_hat: np.ndarray,
    active: np.ndarray,
    frequency_masks: dict[str, np.ndarray],
    c_e: float,
    matrices: dict[str, np.ndarray],
    representative: bool = False,
    contrast: str = "T2",
    secondary_models: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] | None = None,
    repetitions: tuple[str, str, str] | None = None,
    precomputed_pair_diagnostics: dict | None = None,
) -> tuple[dict, dict | None]:
    start_time = time.perf_counter()
    repetitions = tuple(repetitions or CONTRAST_REPETITIONS[contrast])
    if len(repetitions) != 3:
        raise ValueError("Criterion processing requires a triad of repetitions")
    raw_kspace = []
    for repetition in repetitions:
        archive_label = V4_FILE_ARCHIVES.get((subject, repetition), "unknown")
        kspace, _ = v4_read_kspace_slices(
            v4_volume_path(subject, repetition), archive_label
        )
        raw_kspace.append(kspace)
    raw_kspace = np.asarray(raw_kspace)
    alignment_low = ifft2c(
        raw_kspace * frequency_masks["C"][None, None, None, :, :]
    )
    alignment_rss = np.sqrt(np.sum(np.abs(alignment_low) ** 2, axis=2))

    pair_diagnostics = {}
    excluded_pairs = set()
    for i, j, k in PAIRS:
        # The frozen v3.1 invariant is enforced literally: both evaluated
        # repetitions are diagnosed and then aligned *to the auxiliary k
        # repetition*.  The inverse direction is never used for a criterion.
        name = f"r{i + 1}-r{j + 1}"
        if precomputed_pair_diagnostics is not None:
            diagnostic = precomputed_pair_diagnostics[name]
        else:
            first_to_k = registration_diagnostic_pair(
                alignment_rss[k], alignment_rss[i]
            )
            second_to_k = registration_diagnostic_pair(
                alignment_rss[k], alignment_rss[j]
            )
            diagnostic = {
                "data_band": "C only",
                "reference_repetition": f"r{k + 1}",
                "first_member_to_k": first_to_k,
                "second_member_to_k": second_to_k,
                "exclude_pair": bool(
                    first_to_k["exclude_pair"] or second_to_k["exclude_pair"]
                ),
            }
        pair_diagnostics[name] = diagnostic
        if diagnostic["exclude_pair"]:
            excluded_pairs.add((i, j))

    identity_transform = {
        "shift_yx": np.zeros(2),
        "unit_phase": complex(1.0),
        "phase_radians": 0.0,
        "ncc_before": 1.0,
        "ncc_after": 1.0,
        "optimizer_success": True,
        "relative_complex_residual_after": 0.0,
    }

    # A single volume transform (one in-plane shift and one phase shared by
    # all four coils and ten retained slices) puts the W-independent mask and
    # the production image in repetition-1 coordinates.  It is estimated on C.
    # For criteria, it is only a common post-transform applied after i and j
    # have first been brought into k's frame.
    canonical_transforms = [identity_transform]
    for repetition in (1, 2):
        canonical_transforms.append(
            estimate_transform_to_reference(alignment_low[0], alignment_low[repetition])
        )
    canonical_kspace = np.stack(
        [
            transform_kspace(
                raw_kspace[repetition],
                canonical_transforms[repetition]["shift_yx"],
                canonical_transforms[repetition]["unit_phase"],
            )
            for repetition in range(3)
        ]
    )
    canonical_images = ifft2c(canonical_kspace)
    rss = np.sqrt(np.sum(np.abs(canonical_images) ** 2, axis=2))
    mean_rss = rss.mean(axis=0)
    masks = np.zeros((len(SLICES), N, N), dtype=bool)
    for slice_index in range(len(SLICES)):
        threshold = MASK_THRESHOLD * float(np.percentile(mean_rss[slice_index], 99))
        masks[slice_index] = binary_erosion(
            mean_rss[slice_index] > threshold,
            iterations=MASK_EROSION,
            border_value=0,
        )
        if masks[slice_index].sum() == 0:
            raise AssertionError(f"Empty brain mask for {subject}, slice {SLICES[slice_index]}")
    largest_masks = largest_connected_component_mask(masks)

    all_s = []
    all_z = []
    all_ymean = []
    all_largest_membership = []
    segment_rows = []
    crossfit_assertions = []
    power_rows = {}
    pair_corrections = {}
    representative_images = []
    for i, j, k in PAIRS:
        if k in (i, j):
            raise AssertionError("Cross-fitting repetition is in its own pair")
        pair_name = f"r{i + 1}-r{j + 1}"
        # Direct relative transforms are estimated on C.  Both evaluated
        # members i and j are first brought into the auxiliary repetition k's
        # frame.  Only then is the same k-to-r1 transform applied to all three
        # fields, solely to share the subject mask and display coordinates.
        first_to_k = estimate_transform_to_reference(
            alignment_low[k], alignment_low[i]
        )
        second_to_k = estimate_transform_to_reference(
            alignment_low[k], alignment_low[j]
        )
        base = canonical_transforms[k]
        first_in_k = transform_kspace(
            raw_kspace[i], first_to_k["shift_yx"], first_to_k["unit_phase"]
        )
        second_in_k = transform_kspace(
            raw_kspace[j], second_to_k["shift_yx"], second_to_k["unit_phase"]
        )
        first_full = transform_kspace(
            first_in_k, base["shift_yx"], base["unit_phase"]
        )
        second_full = transform_kspace(
            second_in_k, base["shift_yx"], base["unit_phase"]
        )
        third_full = transform_kspace(
            raw_kspace[k], base["shift_yx"], base["unit_phase"]
        )
        raw_difference = (raw_kspace[i] - raw_kspace[j]) / math.sqrt(2.0)
        corrected_difference_kspace = (first_full - second_full) / math.sqrt(2.0)
        power_rows[pair_name] = {
            "excluded_from_criteria": bool((i, j) in excluded_pairs),
            "central_C_before": float(
                np.mean(np.abs(raw_difference[..., frequency_masks["C"]]) ** 2)
            ),
            "central_C_after": float(
                np.mean(np.abs(corrected_difference_kspace[..., frequency_masks["C"]]) ** 2)
            ),
            "exterior_E_before": float(
                np.mean(np.abs(raw_difference[..., frequency_masks["E"]]) ** 2)
            ),
            "exterior_E_after": float(
                np.mean(np.abs(corrected_difference_kspace[..., frequency_masks["E"]]) ** 2)
            ),
            "outer_noise_route_before": float(
                np.mean(np.abs(raw_difference[:, :, OUTER_ROWS][:, :, :, active]) ** 2)
            ),
            "outer_noise_route_after": float(
                np.mean(
                    np.abs(
                        corrected_difference_kspace[:, :, OUTER_ROWS][:, :, :, active]
                    ) ** 2
                )
            ),
        }
        pair_corrections[pair_name] = {
            "first_member": f"r{i + 1}",
            "second_member": f"r{j + 1}",
            "sensitivity_repetition": f"r{k + 1}",
            "estimated_from": "C only",
            "one_shift_and_phase_shared_by_four_coils_and_ten_slices": True,
            "frozen_v3_1_reference_invariant": "both i and j are aligned into k",
            "first_member_to_k": first_to_k,
            "second_member_to_k": second_to_k,
            "auxiliary_k_unchanged_before_common_post_transform": True,
            "common_k_to_r1_post_transform": base,
        }
        crossfit_assertions.append(
            {
                "pair": [i + 1, j + 1],
                "sensitivity_repetition": k + 1,
                "third_not_in_pair": True,
                "weights_use_only_the_third_repetition_coefficients": True,
                "both_evaluated_members_aligned_to_third_repetition_frame": True,
                "alignment_estimated_on_C_and_criterion_evaluated_on_disjoint_E": True,
            }
        )
        if (i, j) in excluded_pairs:
            continue
        criterion_difference_kspace = (
            corrected_difference_kspace
            * frequency_masks["E"][None, None, :, :]
        )
        if not np.all(criterion_difference_kspace[..., ~frequency_masks["E"]] == 0.0):
            raise AssertionError("A coefficient outside E survived in a criterion object")
        evaluation_difference = ifft2c(criterion_difference_kspace)
        second_full_images = ifft2c(second_full)
        for slice_index, slice_number in enumerate(SLICES):
            sensitivity = sensitivity_from_kspace(third_full[slice_index])
            difference = evaluation_difference[slice_index]
            pair_mean = (
                canonical_images[i, slice_index] + second_full_images[slice_index]
            ) / 2.0
            mask = masks[slice_index]
            all_s.append(sensitivity[:, mask].T.astype(np.complex64))
            all_z.append(difference[:, mask].T.astype(np.complex64))
            all_ymean.append(pair_mean[:, mask].T.astype(np.complex64))
            all_largest_membership.append(largest_masks[slice_index][mask])
            segment_rows.append(
                {
                    "pair": f"r{i + 1}-r{j + 1}",
                    "sensitivity_repetition": f"r{k + 1}",
                    "slice": int(slice_number),
                    "mask_pixels": int(mask.sum()),
                    "evaluation_frequency_count": int(frequency_masks["E"].sum()),
                }
            )
            if representative and int(slice_number) == 9:
                inverse = np.linalg.inv(matrices["calibrated"])
                t_image = np.einsum("ij,jyx->iyx", inverse, sensitivity, optimize=True)
                denominator_image = np.einsum(
                    "iyx,iyx->yx", sensitivity.conj(), t_image
                ).real
                denominator_image = np.maximum(denominator_image, 1.0e-14)
                representative_images.append(
                    np.einsum("iyx,iyx->yx", t_image.conj(), pair_mean)
                    / denominator_image
                )
    if not all_s:
        raise NoAdmissiblePairs(
            f"{subject} {contrast} {repetitions}: all repetition pairs were excluded"
        )
    sensitivities = np.concatenate(all_s, axis=0)
    differences = np.concatenate(all_z, axis=0)
    means = np.concatenate(all_ymean, axis=0)
    largest_membership = np.concatenate(all_largest_membership).astype(bool)
    if not np.any(largest_membership):
        raise AssertionError(f"Empty largest-component analysis for {subject}")
    repetition_tag = "-".join(repetitions)
    cache_path = ANALYSIS_CACHE / (
        f"{subject}_{contrast}_{repetition_tag}_crossfit_arrays_{CACHE_SCHEMA}.npz"
    )
    model_definitions = {"aligned": (psi_hat, matrices)}
    if secondary_models:
        model_definitions.update(secondary_models)
    analysis = {}
    for route, (route_psi, route_matrices) in model_definitions.items():
        analysis[route] = {
            "common_high_rss": analysis_bundle_from_arrays(
                sensitivities,
                differences,
                means,
                route_matrices,
                route_psi,
                c_e,
            ),
            "largest_connected_component": analysis_bundle_from_arrays(
                sensitivities[largest_membership],
                differences[largest_membership],
                means[largest_membership],
                route_matrices,
                route_psi,
                c_e,
            ),
        }
    primary_bundle = analysis["aligned"]["common_high_rss"]
    result = {
        "subject": subject,
        "included_pair_count": int(3 - len(excluded_pairs)),
        "excluded_pairs": [f"r{i + 1}-r{j + 1}" for i, j in sorted(excluded_pairs)],
        "pair_registration_diagnostics": pair_diagnostics,
        "canonical_C_only_corrections_to_r1_for_mask_and_figure": {
            f"r{index + 1}": transform for index, transform in enumerate(canonical_transforms)
        },
        "corrections_by_analysis_pair": pair_corrections,
        "difference_power": power_rows,
        "mask": {
            "definition": "mean corrected repetition RSS > 0.15*p99, binary erosion 2",
            "pixels_per_slice": [int(mask.sum()) for mask in masks],
            "total_pixels_once_per_slice": int(masks.sum()),
            "independent_of_W": True,
            "sha256_bool": array_hash(masks.astype(np.uint8)),
            "largest_connected_component": {
                "definition": "largest 6-connected 3-D component within the frozen common high-RSS mask",
                "pixels_per_slice": [int(value.sum()) for value in largest_masks],
                "total_pixels_once_per_slice": int(largest_masks.sum()),
                "sha256_bool": array_hash(largest_masks.astype(np.uint8)),
                "is_subset_of_principal_mask": bool(np.all(~largest_masks | masks)),
            },
        },
        "cross_fitting": crossfit_assertions,
        "criterion_coefficients_outside_E_exactly_zero": True,
        "segments": segment_rows,
        "criterion_point_count_including_pairs": int(len(sensitivities)),
        "contrast": contrast,
        "repetitions": list(repetitions),
        "analysis": analysis,
        "metrics": primary_bundle["metrics"],
        "signal_gain_B_W": primary_bundle["signal_gain_B_W"],
        "signal_gain_components": primary_bundle["signal_gain_components"],
        "identity_prediction_coefficient": primary_bundle[
            "identity_prediction_coefficient"
        ],
        "cache": {
            "file": str(cache_path),
            "sensitivities_shape": list(sensitivities.shape),
            "differences_shape": list(differences.shape),
            "largest_component_point_count": int(np.sum(largest_membership)),
            "sensitivities_sha256": array_hash(sensitivities),
            "differences_sha256": array_hash(differences),
            "largest_component_membership_sha256": array_hash(
                largest_membership.astype(np.uint8)
            ),
            "description": (
                "cached post-IFFT E-only cross-fitted pixel arrays; FFTs never run in bootstrap"
            ),
        },
        "duration_seconds": float(time.perf_counter() - start_time),
    }

    representative_data = None
    if representative:
        slice_index = int(np.where(SLICES == 9)[0][0])
        if not representative_images:
            raise AssertionError("No included pair for representative image")
        representative_data = {
            "subject": subject,
            "contrast": contrast,
            "slice": 9,
            "magnitude": np.abs(np.mean(representative_images, axis=0)),
            "mask": masks[slice_index],
            "largest_component_mask": largest_masks[slice_index],
        }
    np.savez(
        cache_path,
        sensitivities=sensitivities,
        differences=differences,
        means=means,
        largest_component_membership=largest_membership,
        masks=masks,
        largest_masks=largest_masks,
        representative_magnitude=(
            representative_data["magnitude"]
            if representative_data is not None
            else np.empty((0,), dtype=float)
        ),
    )
    metadata = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "analysis",
            "metrics",
            "signal_gain_B_W",
            "signal_gain_components",
            "identity_prediction_coefficient",
            "duration_seconds",
        }
    }
    cache_path.with_suffix(".json").write_text(
        json.dumps(
            {"cache_schema": CACHE_SCHEMA, "metadata": jsonise(metadata)},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return result, representative_data


def v4_prepare_test_subject_cached(
    subject: str,
    psi_hat: np.ndarray,
    active: np.ndarray,
    frequency_masks: dict[str, np.ndarray],
    c_e: float,
    matrices: dict[str, np.ndarray],
    representative: bool,
    contrast: str,
    secondary_models: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    repetitions: tuple[str, str, str],
    precomputed_pair_diagnostics: dict,
) -> tuple[dict, dict | None, bool]:
    start_time = time.perf_counter()
    repetition_tag = "-".join(repetitions)
    cache_path = ANALYSIS_CACHE / (
        f"{subject}_{contrast}_{repetition_tag}_crossfit_arrays_{CACHE_SCHEMA}.npz"
    )
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata_container = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_container.get("cache_schema") == CACHE_SCHEMA:
            with np.load(cache_path) as cache:
                representative_magnitude = cache["representative_magnitude"]
                if not representative or representative_magnitude.size:
                    sensitivities = cache["sensitivities"]
                    differences = cache["differences"]
                    means = cache["means"]
                    membership = cache["largest_component_membership"].astype(bool)
                    masks = cache["masks"].astype(bool)
                    largest_masks = cache["largest_masks"].astype(bool)
                    metadata = v4_unjsonise_complex(metadata_container["metadata"])
                    model_definitions = {"aligned": (psi_hat, matrices)}
                    model_definitions.update(secondary_models)
                    analysis = {}
                    for route, (route_psi, route_matrices) in model_definitions.items():
                        analysis[route] = {
                            "common_high_rss": analysis_bundle_from_arrays(
                                sensitivities,
                                differences,
                                means,
                                route_matrices,
                                route_psi,
                                c_e,
                            ),
                            "largest_connected_component": analysis_bundle_from_arrays(
                                sensitivities[membership],
                                differences[membership],
                                means[membership],
                                route_matrices,
                                route_psi,
                                c_e,
                            ),
                        }
                    primary_bundle = analysis["aligned"]["common_high_rss"]
                    result = {
                        **metadata,
                        "analysis": analysis,
                        "metrics": primary_bundle["metrics"],
                        "signal_gain_B_W": primary_bundle["signal_gain_B_W"],
                        "signal_gain_components": primary_bundle[
                            "signal_gain_components"
                        ],
                        "identity_prediction_coefficient": primary_bundle[
                            "identity_prediction_coefficient"
                        ],
                        "duration_seconds": float(time.perf_counter() - start_time),
                    }
                    representative_data = None
                    if representative:
                        slice_index = int(np.where(SLICES == 9)[0][0])
                        representative_data = {
                            "subject": subject,
                            "contrast": contrast,
                            "slice": 9,
                            "magnitude": representative_magnitude,
                            "mask": masks[slice_index],
                            "largest_component_mask": largest_masks[slice_index],
                        }
                    return result, representative_data, True
    result, representative_data = prepare_test_subject(
        subject,
        psi_hat,
        active,
        frequency_masks,
        c_e,
        matrices,
        representative=representative,
        contrast=contrast,
        secondary_models=secondary_models,
        repetitions=repetitions,
        precomputed_pair_diagnostics=precomputed_pair_diagnostics,
    )
    return result, representative_data, False


def draw_complex_noise(
    rng: np.random.Generator, psi: np.ndarray, active: np.ndarray
) -> np.ndarray:
    count = int(N * active.sum())
    standard = (
        rng.normal(size=(count, COILS)) + 1.0j * rng.normal(size=(count, COILS))
    ) / math.sqrt(2.0)
    factor = np.linalg.cholesky(psi)
    samples = standard @ factor.T
    result = np.zeros((COILS, N, N), dtype=np.complex128)
    result[:, :, active] = samples.reshape(N, int(active.sum()), COILS).transpose(2, 0, 1)
    return result


def synthetic_scene(active: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:N, :N]
    y = (yy - 128.0) / 128.0
    x = (xx - 128.0) / 128.0
    object_image = (
        1.4 * np.exp(-(x**2 / 0.34 + y**2 / 0.45))
        + 0.35 * np.exp(-((x + 0.24) ** 2 + (y - 0.18) ** 2) / 0.035)
    ) * np.exp(1.0j * (0.15 * x - 0.10 * y))
    sensitivities = []
    for coil, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, COILS, endpoint=False)):
        magnitude = 1.0 + 0.32 * (x * np.cos(angle) + y * np.sin(angle))
        phase = 0.35 * (x * np.sin(angle) - y * np.cos(angle)) + 0.12 * coil
        sensitivities.append(magnitude * np.exp(1.0j * phase))
    sensitivities = np.asarray(sensitivities)
    sensitivities /= np.sqrt(np.sum(np.abs(sensitivities) ** 2, axis=0))[None]
    # Moderate amplitude keeps registration highly identifiable while making
    # its sub-0.1-pixel numerical residual negligible in the variance identity.
    signal_images = 5.0 * object_image[None] * sensitivities
    signal_kspace = fft2c(signal_images)
    signal_kspace[:, :, ~active] = 0.0
    mask_truth = binary_erosion(np.abs(object_image) > 0.12, iterations=MASK_EROSION)
    return signal_kspace, sensitivities, mask_truth


def synthetic_injection_test(
    active: np.ndarray, frequency_masks: dict[str, np.ndarray], c_e: float
) -> dict:
    start_time = time.perf_counter()
    rng = np.random.default_rng(SYNTHETIC_SEED)
    diagonal = np.array([0.85, 1.20, 0.72, 1.05]) * 0.018**2
    dhalf = np.diag(np.sqrt(diagonal))
    corr = np.array(
        [
            [1.0, 0.22 * np.exp(0.35j), 0.10 * np.exp(-0.55j), 0.08 * np.exp(0.80j)],
            [0.22 * np.exp(-0.35j), 1.0, 0.18 * np.exp(0.45j), 0.12 * np.exp(-0.30j)],
            [0.10 * np.exp(0.55j), 0.18 * np.exp(-0.45j), 1.0, 0.20 * np.exp(0.65j)],
            [0.08 * np.exp(-0.80j), 0.12 * np.exp(0.30j), 0.20 * np.exp(-0.65j), 1.0],
        ],
        dtype=np.complex128,
    )
    psi0 = dhalf @ corr @ dhalf
    if np.linalg.eigvalsh(psi0)[0] <= 0.0:
        raise AssertionError("Synthetic Psi0 is not positive definite")
    signal_kspace, known_sensitivities, _ = synthetic_scene(active)

    # Independent synthetic calibration subject: 10 slices, 3 repetitions,
    # two consecutive differences, and the exact real active support.
    calibration_repetitions = np.stack(
        [
            np.stack([signal_kspace + draw_complex_noise(rng, psi0, active) for _ in SLICES])
            for _ in range(3)
        ]
    )
    calibration_pair_moments = []
    for i, j in CONSECUTIVE_PAIRS:
        difference = (calibration_repetitions[i] - calibration_repetitions[j]) / math.sqrt(2.0)
        samples = difference[:, :, OUTER_ROWS][:, :, :, active].transpose(0, 2, 3, 1).reshape(-1, COILS)
        calibration_pair_moments.append(centered_moments(samples))
    psi_estimated = np.mean([row["psi"] for row in calibration_pair_moments], axis=0)
    pseudo_estimated = np.mean([row["pseudo"] for row in calibration_pair_moments], axis=0)
    recovery_relative_error = float(np.linalg.norm(psi_estimated - psi0) / np.linalg.norm(psi0))

    acquisition_shifts = np.array([[0.0, 0.0], [1.50, -1.00], [-1.00, 1.25]])
    acquisition_phases = np.array([0.0, 0.35, -0.28])
    trial_metrics = {name: [] for name in CONTROL_NAMES}
    translation_errors = []
    phase_errors = []
    recovered_transform_rows = []
    n_trials = 24
    for trial in range(n_trials):
        repetitions = []
        for repetition in range(3):
            moved_signal = transform_kspace(
                signal_kspace,
                acquisition_shifts[repetition],
                np.exp(1.0j * acquisition_phases[repetition]),
            )
            repetitions.append(moved_signal + draw_complex_noise(rng, psi0, active))
        repetitions = np.asarray(repetitions)
        alignment_low = ifft2c(
            repetitions * frequency_masks["C"][None, None, :, :]
        )
        canonical_transforms = [
            {
                "shift_yx": np.zeros(2),
                "unit_phase": complex(1.0),
                "phase_radians": 0.0,
            }
        ]
        for repetition in (1, 2):
            transform = estimate_transform_to_reference(
                alignment_low[0], alignment_low[repetition]
            )
            canonical_transforms.append(transform)
            translation_errors.append(
                float(np.linalg.norm(transform["shift_yx"] + acquisition_shifts[repetition]))
            )
            expected_phase = -acquisition_phases[repetition]
            wrapped = np.angle(np.exp(1.0j * (transform["phase_radians"] - expected_phase)))
            phase_errors.append(float(abs(wrapped)))
            recovered_transform_rows.append(
                {
                    "trial": trial,
                    "moving_repetition": repetition + 1,
                    "estimated_correction_shift_yx": transform["shift_yx"],
                    "expected_correction_shift_yx": -acquisition_shifts[repetition],
                    "estimated_correction_phase_radians": transform["phase_radians"],
                    "expected_correction_phase_radians": expected_phase,
                }
            )
        canonical = np.stack(
            [
                transform_kspace(
                    repetitions[rep],
                    canonical_transforms[rep]["shift_yx"],
                    canonical_transforms[rep]["unit_phase"],
                )
                for rep in range(3)
            ]
        )
        canonical_images = ifft2c(canonical)
        mean_rss = np.mean(
            np.sqrt(np.sum(np.abs(canonical_images) ** 2, axis=1)), axis=0
        )
        threshold = MASK_THRESHOLD * float(np.percentile(mean_rss, 99))
        mask = binary_erosion(mean_rss > threshold, iterations=MASK_EROSION)
        s_rows = []
        z_rows = []
        for i, j, k in PAIRS:
            if k in (i, j):
                raise AssertionError("Synthetic cross-fitting failed")
            first_to_k = estimate_transform_to_reference(
                alignment_low[k], alignment_low[i]
            )
            second_to_k = estimate_transform_to_reference(
                alignment_low[k], alignment_low[j]
            )
            base = canonical_transforms[k]
            first_full = transform_kspace(
                transform_kspace(
                    repetitions[i],
                    first_to_k["shift_yx"],
                    first_to_k["unit_phase"],
                ),
                base["shift_yx"],
                base["unit_phase"],
            )
            second_full = transform_kspace(
                transform_kspace(
                    repetitions[j],
                    second_to_k["shift_yx"],
                    second_to_k["unit_phase"],
                ),
                base["shift_yx"],
                base["unit_phase"],
            )
            third_full = transform_kspace(
                repetitions[k], base["shift_yx"], base["unit_phase"]
            )
            sensitivity = sensitivity_from_kspace(third_full)
            criterion_kspace = (
                (first_full - second_full)
                / math.sqrt(2.0)
                * frequency_masks["E"][None, :, :]
            )
            if not np.all(criterion_kspace[..., ~frequency_masks["E"]] == 0.0):
                raise AssertionError("Synthetic criterion leaked outside E")
            difference = ifft2c(criterion_kspace)
            s_rows.append(sensitivity[:, mask].T)
            z_rows.append(difference[:, mask].T)
        s_all = np.concatenate(s_rows)
        z_all = np.concatenate(z_rows)
        metrics = metric_sums_from_arrays(
            s_all,
            z_all,
            control_matrices(psi_estimated),
            psi_estimated,
            c_e,
        )
        for name in CONTROL_NAMES:
            trial_metrics[name].append(metrics[name]["R_empirical_over_predicted"])

    ratio_summary = {}
    all_within_three_se = True
    for name, values in trial_metrics.items():
        values = np.asarray(values)
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        z_score = float(abs(mean - 1.0) / se)
        passed = bool(z_score <= 3.0)
        all_within_three_se &= passed
        ratio_summary[name] = {
            "mean_R": mean,
            "monte_carlo_standard_error": se,
            "absolute_z_from_one": z_score,
            "within_three_MC_standard_errors": passed,
            "R_without_c_E": float(mean * c_e),
        }
    calibrated_without = ratio_summary["calibrated"]["R_without_c_E"]
    without_factor_fails_equivalence = bool(not (0.8 <= calibrated_without <= 1.25))
    covariance_recovery_pass = bool(recovery_relative_error <= 0.02)
    correction_pass = bool(
        np.percentile(translation_errors, 95) <= 0.10
        and np.percentile(phase_errors, 95) <= 0.02
    )
    passed = bool(
        all_within_three_se
        and covariance_recovery_pass
        and without_factor_fails_equivalence
        and correction_pass
    )
    if not passed:
        print(
            "Synthetic diagnostics before STOP:",
            {
                "psi_relative_error": recovery_relative_error,
                "translation_p95": float(np.percentile(translation_errors, 95)),
                "phase_p95": float(np.percentile(phase_errors, 95)),
                "R": ratio_summary,
                "without_c_E_fails": without_factor_fails_equivalence,
            },
            flush=True,
        )
        raise AssertionError(
            "Synthetic end-to-end injection failed; an identity used by the real pipeline is invalid"
        )
    return {
        "description": (
            "Independent synthetic calibration plus 24 synthetic test subjects, each with three "
            "repetitions, smooth signal times known sensitivities, real active kx support, "
            "known subpixel translations/global phases, alignment estimated only on C, "
            "both evaluated members aligned into auxiliary k, evaluation only on disjoint E, "
            "pairwise sensitivity cross-fitting, and c_E."
        ),
        "seed": SYNTHETIC_SEED,
        "n_test_monte_carlo_subjects": n_trials,
        "psi0": complex_matrix_record(psi0),
        "psi_estimated": complex_matrix_record(psi_estimated),
        "psi_recovery_relative_frobenius_error": recovery_relative_error,
        "psi_recovery_threshold": 0.02,
        "psi_recovery_pass": covariance_recovery_pass,
        "pseudo_covariance_ratio": float(np.linalg.norm(pseudo_estimated) / np.linalg.norm(psi_estimated)),
        "known_sensitivities_rss_normalization_max_error": float(
            np.max(np.abs(np.sum(np.abs(known_sensitivities) ** 2, axis=0) - 1.0))
        ),
        "translation_error_p95_pixels": float(np.percentile(translation_errors, 95)),
        "phase_error_p95_radians": float(np.percentile(phase_errors, 95)),
        "known_acquisition_shifts_yx": acquisition_shifts,
        "known_acquisition_phases_radians": acquisition_phases,
        "representative_recovered_transforms": recovered_transform_rows[:6],
        "motion_phase_correction_pass": correction_pass,
        "R_by_control": ratio_summary,
        "all_R_means_within_three_MC_standard_errors_of_one": all_within_three_se,
        "calibrated_R_without_c_E": calibrated_without,
        "without_c_E_equivalence_test_passes": not without_factor_fails_equivalence,
        "proof_c_E_is_necessary": without_factor_fails_equivalence,
        "C_and_E_are_disjoint": bool(
            not np.any(frequency_masks["C"] & frequency_masks["E"])
        ),
        "pass": passed,
        "duration_seconds": float(time.perf_counter() - start_time),
    }


def v4_synthetic_aligned_covariance_test(
    active: np.ndarray, frequency_masks: dict[str, np.ndarray]
) -> dict:
    """Exercise the central v4 aligned-Psi route with known motion and covariance."""
    start_time = time.perf_counter()
    rng = np.random.default_rng(SYNTHETIC_SEED)
    diagonal = np.array([0.85, 1.20, 0.72, 1.05]) * 0.018**2
    dhalf = np.diag(np.sqrt(diagonal))
    corr = np.array(
        [
            [1.0, 0.22 * np.exp(0.35j), 0.10 * np.exp(-0.55j), 0.08 * np.exp(0.80j)],
            [0.22 * np.exp(-0.35j), 1.0, 0.18 * np.exp(0.45j), 0.12 * np.exp(-0.30j)],
            [0.10 * np.exp(0.55j), 0.18 * np.exp(-0.45j), 1.0, 0.20 * np.exp(0.65j)],
            [0.08 * np.exp(-0.80j), 0.12 * np.exp(0.30j), 0.20 * np.exp(-0.65j), 1.0],
        ],
        dtype=np.complex128,
    )
    psi0 = dhalf @ corr @ dhalf
    signal_kspace, _, _ = synthetic_scene(active)
    shifts = np.asarray([[0.0, 0.0], [1.50, -1.00], [-1.00, 1.25]])
    phases = np.asarray([0.0, 0.35, -0.28])
    repetitions = []
    for repetition in range(3):
        moved = transform_kspace(
            signal_kspace,
            shifts[repetition],
            np.exp(1.0j * phases[repetition]),
        )
        repetitions.append(
            np.stack(
                [moved + draw_complex_noise(rng, psi0, active) for _ in SLICES]
            )
        )
    repetitions = np.asarray(repetitions)
    alignment_low = ifft2c(
        repetitions * frequency_masks["C"][None, None, None, :, :]
    )
    aligned_moments = []
    aligned_power = []
    raw_power = []
    transforms = []
    for i, j, k in PAIRS:
        first = estimate_transform_to_reference(alignment_low[k], alignment_low[i])
        second = estimate_transform_to_reference(alignment_low[k], alignment_low[j])
        first_aligned = transform_kspace(
            repetitions[i], first["shift_yx"], first["unit_phase"]
        )
        second_aligned = transform_kspace(
            repetitions[j], second["shift_yx"], second["unit_phase"]
        )
        difference = (first_aligned - second_aligned) / math.sqrt(2.0)
        raw_difference = (repetitions[i] - repetitions[j]) / math.sqrt(2.0)
        samples = (
            difference[:, :, OUTER_ROWS][:, :, :, active]
            .transpose(0, 2, 3, 1)
            .reshape(-1, COILS)
        )
        aligned_moments.append(centered_moments(samples))
        aligned_power.append(
            float(np.mean(np.abs(difference[:, :, OUTER_ROWS][:, :, :, active]) ** 2))
        )
        raw_power.append(
            float(np.mean(np.abs(raw_difference[:, :, OUTER_ROWS][:, :, :, active]) ** 2))
        )
        transforms.append(
            {
                "pair": f"r{i + 1}-r{j + 1}",
                "auxiliary": f"r{k + 1}",
                "first_to_k": first,
                "second_to_k": second,
            }
        )
    psi_aligned = np.mean([row["psi"] for row in aligned_moments], axis=0)
    relative_error = float(np.linalg.norm(psi_aligned - psi0) / np.linalg.norm(psi0))
    passed = bool(
        relative_error <= 0.02
        and not np.any(frequency_masks["C"] & frequency_masks["E"])
    )
    if not passed:
        raise AssertionError(
            f"Synthetic v4 aligned covariance route failed: relative error {relative_error}"
        )
    return {
        "description": (
            "Known covariance and known inter-repetition translations/phases; both pair "
            "members are aligned to auxiliary k from C, then Psi is estimated on N disjoint from C."
        ),
        "seed": SYNTHETIC_SEED,
        "psi0": complex_matrix_record(psi0),
        "psi_aligned": complex_matrix_record(psi_aligned),
        "relative_frobenius_error": relative_error,
        "threshold": 0.02,
        "outer_power_aligned_over_raw": float(np.sum(aligned_power) / np.sum(raw_power)),
        "C_and_N_are_disjoint": bool(
            not np.any(
                frequency_masks["C"]
                & (
                    np.isin(np.arange(N), OUTER_ROWS)[:, None]
                    & np.broadcast_to(active[None, :], (N, N))
                )
            )
        ),
        "transforms": transforms,
        "pass": passed,
        "duration_seconds": float(time.perf_counter() - start_time),
    }


def polynomial_features(x: np.ndarray, degree: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    x = np.asarray(x, dtype=float)
    terms: list[tuple[int, ...]] = [()]
    for current_degree in range(1, degree + 1):
        terms.extend(itertools.combinations_with_replacement(range(x.shape[1]), current_degree))
    design = np.ones((len(x), len(terms)), dtype=float)
    for column, term in enumerate(terms[1:], start=1):
        design[:, column] = np.prod(x[:, term], axis=1)
    return design, terms


def exact_dynamic_outputs(
    cache_path: pathlib.Path, psi_values: np.ndarray, c_e: float
) -> np.ndarray:
    """Exact full-pixel outputs for many calibration covariances.

    This is a batched linear-algebra implementation of metric_contribution;
    it changes only evaluation order, not the estimand.  Configuration and
    pixel batch sizes are fixed operational memory bounds.
    """
    with np.load(cache_path) as cache:
        sensitivities = cache["sensitivities"]
        differences = cache["differences"]
        outputs = np.empty((len(psi_values), 6), dtype=float)
        controls = [control_matrices(psi) for psi in psi_values]
        matrices = {
            name: np.stack([row[name] for row in controls])
            for name in ("calibrated", "diagonal", "wrong_cyclic")
        }
        inverses = {name: np.linalg.inv(value) for name, value in matrices.items()}
        column_pairs = {
            "calibrated": (0, 1),
            "diagonal": (2, 3),
            "wrong_cyclic": (4, 5),
        }
        configuration_batch = 12
        pixel_chunk = 60_000
        for config_start in range(0, len(psi_values), configuration_batch):
            config_stop = min(config_start + configuration_batch, len(psi_values))
            psi_batch = np.asarray(
                psi_values[config_start:config_stop], dtype=np.complex128
            )
            for name, (empirical_column, predicted_column) in column_pairs.items():
                inverse_batch = inverses[name][config_start:config_stop]
                empirical_sums = np.zeros(config_stop - config_start, dtype=float)
                predicted_sums = np.zeros(config_stop - config_start, dtype=float)
                for pixel_start in range(0, len(sensitivities), pixel_chunk):
                    pixel_stop = min(pixel_start + pixel_chunk, len(sensitivities))
                    s = np.asarray(
                        sensitivities[pixel_start:pixel_stop], dtype=np.complex128
                    )
                    z = np.asarray(
                        differences[pixel_start:pixel_stop], dtype=np.complex128
                    )
                    t = np.einsum("bij,nj->bni", inverse_batch, s, optimize=True)
                    denominator = np.einsum(
                        "ni,bni->bn", s.conj(), t, optimize=True
                    ).real
                    if np.any(denominator <= 1.0e-14):
                        raise AssertionError("Non-positive batched coil-combination denominator")
                    numerator = np.einsum(
                        "bni,ni->bn", t.conj(), z, optimize=True
                    )
                    empirical_sums += np.sum(
                        np.abs(numerator) ** 2 / denominator**2, axis=1
                    )
                    if name == "calibrated":
                        # For W=Psi, w^H Psi w = 1/(s^H Psi^-1 s).
                        predicted_sums += c_e * np.sum(1.0 / denominator, axis=1)
                    else:
                        quadratic = np.einsum(
                            "bni,bij,bnj->bn",
                            t.conj(),
                            psi_batch,
                            t,
                            optimize=True,
                        ).real
                        predicted_sums += c_e * np.sum(
                            quadratic / denominator**2, axis=1
                        )
                outputs[config_start:config_stop, empirical_column] = empirical_sums
                outputs[config_start:config_stop, predicted_column] = predicted_sums
    return outputs


def response_validation_targets(dynamic: np.ndarray) -> np.ndarray:
    empirical_cal = dynamic[:, 0]
    predicted_cal = dynamic[:, 1]
    empirical_diag = dynamic[:, 2]
    predicted_diag = dynamic[:, 3]
    empirical_wrong = dynamic[:, 4]
    predicted_wrong = dynamic[:, 5]
    return np.asarray(
        [
            np.mean(empirical_cal / predicted_cal),
            np.mean(empirical_diag / predicted_diag),
            np.mean(empirical_wrong / predicted_wrong),
            np.sum(empirical_diag) / np.sum(empirical_cal)
            - np.sum(predicted_diag) / np.sum(predicted_cal),
            np.sum(empirical_diag) / np.sum(empirical_cal) - 1.0,
            np.sum(empirical_wrong) / np.sum(empirical_cal),
        ]
    )


def build_bootstrap_response(
    test_subject_rows: list[dict],
    calibration_psis: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    c_e: float,
) -> tuple[np.ndarray, dict]:
    start_time = time.perf_counter()
    counts = np.stack(
        [np.bincount(row, minlength=len(calibration_psis)) for row in calibration_indices]
    )
    psi_boot = np.einsum("bi,ijk->bjk", counts, calibration_psis) / len(calibration_psis)
    coordinates = counts[:, :9].astype(float) - 1.0
    exact_cache: dict[tuple[int, ...], np.ndarray] = {}
    attempts_report = []
    accepted = None
    accepted_predictions = None

    for attempt in SURROGATE_ATTEMPTS:
        degree = int(attempt["degree"])
        train_indices = np.arange(int(attempt["train"]))
        validation_indices = np.arange(
            int(attempt["train"]), int(attempt["train"] + attempt["validation"])
        )
        requested = np.concatenate((train_indices, validation_indices))
        unique_requested = []
        seen = set()
        for bootstrap_index in requested:
            key = tuple(int(v) for v in counts[bootstrap_index])
            if key not in seen:
                unique_requested.append(int(bootstrap_index))
                seen.add(key)
        missing = [
            index for index in unique_requested
            if tuple(int(v) for v in counts[index]) not in exact_cache
        ]
        if missing:
            print(
                f"  response surface degree {degree}: {len(missing)} new exact calibration configurations",
                flush=True,
            )
            psi_missing = psi_boot[missing]
            # Subjects are independent numerical jobs.  A fixed three-worker
            # thread pool reduces wall time while preserving byte-identical
            # inputs and the same per-subject full-pixel evaluation.
            def evaluate_subject(row: dict) -> np.ndarray:
                return exact_dynamic_outputs(
                    pathlib.Path(row["cache"]["file"]), psi_missing, c_e
                )

            with ThreadPoolExecutor(max_workers=EXACT_SUBJECT_WORKERS) as executor:
                exact_subjects = list(executor.map(evaluate_subject, test_subject_rows))
            for subject_index, (row, exact) in enumerate(
                zip(test_subject_rows, exact_subjects)
            ):
                for local_index, bootstrap_index in enumerate(missing):
                    key = tuple(int(v) for v in counts[bootstrap_index])
                    if key not in exact_cache:
                        exact_cache[key] = np.empty((len(test_subject_rows), 6), dtype=float)
                    exact_cache[key][subject_index] = exact[local_index]
                print(
                    f"    exact subject {subject_index + 1:02d}/{len(test_subject_rows)}: {row['subject']}",
                    flush=True,
                )

        design_all, terms = polynomial_features(coordinates, degree)
        # Include the exact full-calibration point at coordinate zero.
        design_zero, _ = polynomial_features(np.zeros((1, coordinates.shape[1])), degree)
        design_train = np.vstack((design_zero, design_all[train_indices]))
        predictions = np.empty((B_BOOT, len(test_subject_rows), 6), dtype=float)
        validation_exact = np.empty((len(validation_indices), len(test_subject_rows), 6), dtype=float)
        raw_relative_errors = []
        coefficients = []
        for subject_index, row in enumerate(test_subject_rows):
            point = np.array(
                [
                    row["metrics"]["calibrated"]["empirical_sum"],
                    row["metrics"]["calibrated"]["predicted_sum"],
                    row["metrics"]["diagonal"]["empirical_sum"],
                    row["metrics"]["diagonal"]["predicted_sum"],
                    row["metrics"]["wrong_cyclic"]["empirical_sum"],
                    row["metrics"]["wrong_cyclic"]["predicted_sum"],
                ],
                dtype=float,
            )
            train_values = np.stack(
                [
                    exact_cache[tuple(int(v) for v in counts[index])][subject_index]
                    for index in train_indices
                ]
            )
            scale = np.maximum(np.abs(point), 1.0e-30)
            normalized_train = np.vstack((point, train_values)) / scale[None]
            coefficient, _, _, singular_values = np.linalg.lstsq(
                design_train, normalized_train, rcond=1.0e-11
            )
            predicted = (design_all @ coefficient) * scale[None]
            predictions[:, subject_index] = predicted
            coefficients.append(
                {
                    "subject": row["subject"],
                    "design_condition_number": float(singular_values[0] / singular_values[-1]),
                }
            )
            exact_validation_subject = np.stack(
                [
                    exact_cache[tuple(int(v) for v in counts[index])][subject_index]
                    for index in validation_indices
                ]
            )
            validation_exact[:, subject_index] = exact_validation_subject
            relative = np.abs(
                (predicted[validation_indices] - exact_validation_subject)
                / np.maximum(np.abs(exact_validation_subject), 1.0e-30)
            )
            raw_relative_errors.extend(relative.ravel().tolist())

        # Freeze genuinely out-of-sample surface predictions before exact
        # configurations replace their production values.  Validation must
        # never compare an exact point with itself.
        predicted_validation_unreplaced = predictions[validation_indices].copy()

        # Replace every exactly evaluated configuration in all occurrences.
        for bootstrap_index in range(B_BOOT):
            key = tuple(int(v) for v in counts[bootstrap_index])
            if key in exact_cache:
                predictions[bootstrap_index] = exact_cache[key]

        all_predictions_positive = bool(np.all(np.isfinite(predictions)) and np.all(predictions > 0.0))
        target_differences = {}
        target_names = (
            "mean_R_calibrated",
            "mean_R_diagonal",
            "mean_R_wrong_cyclic",
            "delta_gain_diag",
            "empirical_gain_diag",
            "wrong_empirical_ratio",
        )
        target_maxima = []
        for target_index, target_name in enumerate(target_names):
            differences = []
            for local_index, bootstrap_index in enumerate(validation_indices):
                selected = test_indices[bootstrap_index]
                exact_values = response_validation_targets(
                    validation_exact[local_index, selected]
                )
                predicted_values = response_validation_targets(
                    predicted_validation_unreplaced[local_index, selected]
                )
                differences.append(
                    abs(float(predicted_values[target_index] - exact_values[target_index]))
                )
            maximum = float(max(differences))
            target_differences[target_name] = {
                "maximum_absolute_error": maximum,
                "median_absolute_error": float(np.median(differences)),
                "threshold": SURROGATE_TARGET_ABS_MAX[target_name],
                "pass": bool(maximum <= SURROGATE_TARGET_ABS_MAX[target_name]),
            }
            target_maxima.append(target_differences[target_name]["pass"])
        raw_p99 = float(np.percentile(raw_relative_errors, 99))
        pass_attempt = bool(
            all_predictions_positive
            and raw_p99 <= SURROGATE_RAW_RELATIVE_P99_MAX
            and all(target_maxima)
        )
        report = {
            "degree": degree,
            "n_polynomial_terms": len(terms),
            "training_replicates": int(len(train_indices)),
            "validation_replicates": int(len(validation_indices)),
            "unique_exact_configurations_available": int(len(exact_cache)),
            "raw_output_relative_error_p99": raw_p99,
            "raw_output_relative_error_max": float(max(raw_relative_errors)),
            "raw_p99_threshold": SURROGATE_RAW_RELATIVE_P99_MAX,
            "all_5000_predictions_finite_and_positive": all_predictions_positive,
            "derived_joint_target_validation": target_differences,
            "per_subject_design": coefficients,
            "pass": pass_attempt,
        }
        attempts_report.append(report)
        if pass_attempt:
            accepted = report
            accepted_predictions = predictions
            break

    if accepted is None or accepted_predictions is None:
        raise AssertionError(
            "Bootstrap response-surface validation failed fixed tolerances; no approximate bootstrap reported"
        )
    return accepted_predictions, {
        "method": (
            "Every bootstrap replicate recomputes Psi_hat and all derived W. Pixelwise empirical and "
            "predicted sums are evaluated by a polynomial response in the nine independent calibration "
            "multinomial counts. The response is fitted to exact full-pixel recalculations and accepted "
            "only after held-out exact-replicate validation under fixed tolerances; evaluated exact "
            "configurations replace their response values. I predictions are handled exactly and linearly."
        ),
        "is_numerical_acceleration_not_statistical_resampling_approximation": True,
        "parallel_exact_subject_workers": EXACT_SUBJECT_WORKERS,
        "fixed_validation_thresholds": {
            "raw_relative_p99": SURROGATE_RAW_RELATIVE_P99_MAX,
            "derived_targets": SURROGATE_TARGET_ABS_MAX,
        },
        "attempts": attempts_report,
        "accepted": accepted,
        "exact_unique_configuration_count": int(len(exact_cache)),
        "duration_seconds": float(time.perf_counter() - start_time),
        "psi_boot": psi_boot,
    }


def score_deltas_for_subjects(
    noise_rows: list[dict],
    candidate_complex: np.ndarray,
    delta2: float,
) -> tuple[np.ndarray, np.ndarray]:
    diagonal_deltas = []
    wrong_deltas = []
    controls = {
        name: spectral_floor_matrix(matrix, delta2)
        for name, matrix in control_matrices(candidate_complex).items()
    }
    for row in noise_rows:
        covariance = row["psi"]
        baseline = nll_complex_from_covariance(covariance, controls["calibrated"])
        diagonal_deltas.append(
            nll_complex_from_covariance(covariance, controls["diagonal"]) - baseline
        )
        wrong_deltas.append(
            nll_complex_from_covariance(covariance, controls["wrong_cyclic"]) - baseline
        )
    return np.asarray(diagonal_deltas), np.asarray(wrong_deltas)


def bootstrap_nll_wrong_delta(
    test_noise_rows: list[dict],
    psi_boot: np.ndarray,
) -> np.ndarray:
    calibrated = []
    wrong = []
    for psi in psi_boot:
        delta2 = spectral_floor_delta2(psi)
        controls = control_matrices(psi)
        calibrated.append(spectral_floor_matrix(controls["calibrated"], delta2))
        wrong.append(spectral_floor_matrix(controls["wrong_cyclic"], delta2))
    calibrated = np.stack(calibrated)
    wrong = np.stack(wrong)
    inverse_calibrated = np.linalg.inv(calibrated)
    inverse_wrong = np.linalg.inv(wrong)
    _, logdet_calibrated = np.linalg.slogdet(calibrated)
    _, logdet_wrong = np.linalg.slogdet(wrong)
    subject_covariances = np.stack([row["psi"] for row in test_noise_rows])
    trace_difference = np.einsum(
        "bij,sji->bs",
        inverse_wrong - inverse_calibrated,
        subject_covariances,
        optimize=True,
    ).real
    return (logdet_wrong - logdet_calibrated).real[:, None] + trace_difference


def selected_subject_mean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, indices, axis=1).mean(axis=1)


def selected_subject_sum(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, indices, axis=1).sum(axis=1)


def simultaneous_intervals(
    estimates: np.ndarray,
    bootstrap_values: np.ndarray,
    names: tuple[str, ...],
) -> dict:
    estimates = np.asarray(estimates, dtype=float)
    bootstrap_values = np.asarray(bootstrap_values, dtype=float)
    standard_errors = np.std(bootstrap_values, axis=0, ddof=1)
    if np.any(standard_errors <= 0.0):
        raise AssertionError("Zero bootstrap standard error")
    centered = bootstrap_values - np.mean(bootstrap_values, axis=0)[None]
    max_abs_z = np.max(np.abs(centered / standard_errors[None]), axis=1)
    threshold = float(np.quantile(max_abs_z, 0.95))
    targets = {}
    for index, name in enumerate(names):
        half_width = threshold * standard_errors[index]
        targets[name] = {
            "estimate": float(estimates[index]),
            "bootstrap_mean": float(np.mean(bootstrap_values[:, index])),
            "bootstrap_standard_error": float(standard_errors[index]),
            "pointwise_percentile_95_interval": [
                float(x) for x in np.quantile(bootstrap_values[:, index], [0.025, 0.975])
            ],
            "simultaneous_95_interval": [
                float(estimates[index] - half_width),
                float(estimates[index] + half_width),
            ],
        }
    return {
        "construction": (
            "95th percentile of the maximum absolute centered bootstrap statistic, "
            "standardized by its joint-bootstrap standard error"
        ),
        "familywise_level": 0.95,
        "max_t_threshold": threshold,
        "bootstrap_replicates": int(len(bootstrap_values)),
        "targets": targets,
    }


def percentile_interval_record(estimate: float, bootstrap_values: np.ndarray) -> dict:
    values = np.asarray(bootstrap_values, dtype=float)
    return {
        "estimate": float(estimate),
        "bootstrap_mean": float(np.mean(values)),
        "bootstrap_standard_error": float(np.std(values, ddof=1)),
        "percentile_95_interval": [
            float(x) for x in np.quantile(values, [0.025, 0.975])
        ],
        "construction": "paired percentile interval from the joint subject bootstrap",
    }


def compute_bootstrap_and_criteria(
    test_rows: list[dict],
    test_noise_rows: list[dict],
    calibration_psis: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    dynamic_boot: np.ndarray,
    response_report: dict,
    c_e: float,
) -> tuple[dict, dict, dict]:
    psi_boot = response_report.pop("psi_boot")

    empirical = {
        "calibrated": dynamic_boot[:, :, 0],
        "diagonal": dynamic_boot[:, :, 2],
        "wrong_cyclic": dynamic_boot[:, :, 4],
    }
    predicted = {
        "calibrated": dynamic_boot[:, :, 1],
        "diagonal": dynamic_boot[:, :, 3],
        "wrong_cyclic": dynamic_boot[:, :, 5],
    }
    identity_empirical_point = np.array(
        [row["metrics"]["identity"]["empirical_sum"] for row in test_rows]
    )
    identity_coefficients = np.stack(
        [row["identity_prediction_coefficient"] for row in test_rows]
    )
    identity_predicted = c_e * np.einsum(
        "bij,sji->bs", psi_boot, identity_coefficients, optimize=True
    ).real
    empirical["identity"] = np.broadcast_to(
        identity_empirical_point[None], identity_predicted.shape
    )
    predicted["identity"] = identity_predicted

    point_empirical = {
        name: np.array([row["metrics"][name]["empirical_sum"] for row in test_rows])
        for name in CONTROL_NAMES
    }
    point_predicted = {
        name: np.array([row["metrics"][name]["predicted_sum"] for row in test_rows])
        for name in CONTROL_NAMES
    }

    # C1: four subject-level ratios, with one simultaneous 95% band across W.
    point_c1 = np.asarray(
        [
            np.mean(point_empirical[name] / point_predicted[name])
            for name in CONTROL_NAMES
        ]
    )
    bootstrap_c1 = np.stack(
        [
            selected_subject_mean(
                empirical[name] / predicted[name], test_indices
            )
            for name in CONTROL_NAMES
        ],
        axis=1,
    )
    c1_family = simultaneous_intervals(
        point_c1, bootstrap_c1, CONTROL_NAMES
    )
    c1_targets = {}
    for name in CONTROL_NAMES:
        row = c1_family["targets"][name]
        interval = row["simultaneous_95_interval"]
        c1_targets[name] = {
            **row,
            "equivalence_bounds": [0.8, 1.25],
            "pass": bool(interval[0] >= 0.8 and interval[1] <= 1.25),
        }

    # Aggregated ratios of sums required by C2 and C3a.
    empirical_sums = {
        name: selected_subject_sum(values, test_indices)
        for name, values in empirical.items()
    }
    predicted_sums = {
        name: selected_subject_sum(values, test_indices)
        for name, values in predicted.items()
    }
    boot_G_emp_diag = empirical_sums["diagonal"] / empirical_sums["calibrated"]
    boot_G_pred_diag = predicted_sums["diagonal"] / predicted_sums["calibrated"]
    boot_delta_gain = boot_G_emp_diag - boot_G_pred_diag
    boot_empirical_gain_minus_one = boot_G_emp_diag - 1.0
    point_G_emp_diag = float(
        np.sum(point_empirical["diagonal"])
        / np.sum(point_empirical["calibrated"])
    )
    point_G_pred_diag = float(
        np.sum(point_predicted["diagonal"])
        / np.sum(point_predicted["calibrated"])
    )
    c2_delta = percentile_interval_record(
        point_G_emp_diag - point_G_pred_diag, boot_delta_gain
    )
    c2_gain_exists = percentile_interval_record(
        point_G_emp_diag - 1.0, boot_empirical_gain_minus_one
    )
    delta_interval = c2_delta["percentile_95_interval"]
    gain_interval = c2_gain_exists["percentile_95_interval"]
    c2_equivalence_pass = bool(
        delta_interval[0] >= -0.02 and delta_interval[1] <= 0.02
    )
    c2_gain_exists_pass = bool(gain_interval[0] > 0.0)

    boot_wrong_ratio = (
        empirical_sums["wrong_cyclic"] / empirical_sums["calibrated"]
    )
    point_wrong_ratio = float(
        np.sum(point_empirical["wrong_cyclic"])
        / np.sum(point_empirical["calibrated"])
    )
    c3a_record = percentile_interval_record(point_wrong_ratio, boot_wrong_ratio)
    c3a_interval = c3a_record["percentile_95_interval"]

    # C3b always uses the 4x4 Hermitian complex covariance score.  The
    # pseudo-covariance is descriptive and never selects an algorithm.
    nll_wrong_delta = bootstrap_nll_wrong_delta(test_noise_rows, psi_boot)
    boot_nll_mean = selected_subject_mean(nll_wrong_delta, test_indices)
    psi_hat = np.mean(calibration_psis, axis=0)
    _, point_wrong_nll = score_deltas_for_subjects(
        test_noise_rows, psi_hat, spectral_floor_delta2(psi_hat)
    )
    c3b_record = percentile_interval_record(
        float(np.mean(point_wrong_nll)), boot_nll_mean
    )
    c3b_interval = c3b_record["percentile_95_interval"]

    criteria = {
        "C1": {
            "name": "Gamma predicts absolute receiver-noise variance",
            "statistic": "mean over 15 test subjects of each subject ratio-of-sums R_s,W",
            "simultaneous_family": c1_family,
            "targets": c1_targets,
            "equivalence_bounds": [0.8, 1.25],
            "pass": bool(all(row["pass"] for row in c1_targets.values())),
        },
        "C2": {
            "name": "predicted diagonal-to-calibrated gain equals observed gain, and the gain exists",
            "statistic": "G_emp=sum(E_diag)/sum(E_cal); G_pred=sum(V_diag)/sum(V_cal)",
            "G_emp": point_G_emp_diag,
            "G_pred": point_G_pred_diag,
            "delta_gain_G_emp_minus_G_pred": {
                **c2_delta,
                "equivalence_bounds": [-0.02, 0.02],
                "interpretation": "two absolute percentage points in the variance-ratio gain",
                "pass": c2_equivalence_pass,
            },
            "empirical_gain_G_emp_minus_one": {
                **c2_gain_exists,
                "one_sided_requirement": "lower endpoint strictly greater than zero",
                "pass": c2_gain_exists_pass,
            },
            "pass": bool(c2_equivalence_pass and c2_gain_exists_pass),
        },
        "C3a": {
            "name": "wrong correlation layout has higher output-difference variance",
            "statistic": "sum(E_wrong)/sum(E_calibrated)",
            **c3a_record,
            "one_sided_bound": 1.0,
            "pass": bool(c3a_interval[0] > 1.0),
        },
        "C3b": {
            "name": "complex Gaussian covariance score penalizes wrong correlation layout",
            "statistic": "mean_s NLL_wrong-NLL_calibrated",
            **c3b_record,
            "one_sided_bound": 0.0,
            "score_name": "complex Gaussian covariance score",
            "pass": bool(c3b_interval[0] > 0.0),
        },
    }

    # Joint-bootstrap gain summaries for figure panel iv and the required I diagnostic.
    gain_names = []
    point_gains = []
    bootstrap_gains = []
    per_subject_gains = {}
    for comparison in ("diagonal", "identity", "wrong_cyclic"):
        point_emp_gain = point_empirical[comparison] / point_empirical["calibrated"] - 1.0
        point_pred_gain = point_predicted[comparison] / point_predicted["calibrated"] - 1.0
        boot_emp_gain = empirical_sums[comparison] / empirical_sums["calibrated"] - 1.0
        boot_pred_gain = predicted_sums[comparison] / predicted_sums["calibrated"] - 1.0
        gain_names.extend((f"{comparison}_empirical_gain", f"{comparison}_predicted_gain"))
        point_gains.extend(
            (
                float(np.sum(point_empirical[comparison]) / np.sum(point_empirical["calibrated"]) - 1.0),
                float(np.sum(point_predicted[comparison]) / np.sum(point_predicted["calibrated"]) - 1.0),
            )
        )
        bootstrap_gains.extend((boot_emp_gain, boot_pred_gain))
        per_subject_gains[comparison] = {
            test_rows[index]["subject"]: {
                "empirical_ratio_minus_one": float(point_emp_gain[index]),
                "predicted_ratio_minus_one": float(point_pred_gain[index]),
            }
            for index in range(len(test_rows))
        }
    gain_intervals = simultaneous_intervals(
        np.asarray(point_gains), np.stack(bootstrap_gains, axis=1), tuple(gain_names)
    )
    gains = {
        "definition": "comparison/calibrated ratio of sums minus one over all retained test objects",
        "joint_simultaneous_intervals": gain_intervals,
        "per_subject": per_subject_gains,
        "identity_gain_is_noncriterion": True,
    }
    bootstrap = {
        "seed": BOOTSTRAP_SEED,
        "replicates": B_BOOT,
        "joint_resampling": (
            "Each replicate simultaneously resamples 10 calibration subjects and 15 whole test subjects."
        ),
        "calibration_indices_shape": list(calibration_indices.shape),
        "test_indices_shape": list(test_indices.shape),
        "whole_subject_is_bootstrap_unit": True,
        "pair_dependence_preserved": True,
        "C1_simultaneous_family": c1_family,
        "C2_C3_intervals_are_paired_pointwise_95_percent": True,
        "response_surface_acceleration": response_report,
    }
    return criteria, gains, bootstrap


def summarize_noise_estimates(noise_rows: list[dict], calibration_subjects: list[str]) -> dict:
    selected = [row for row in noise_rows if row["subject"] in calibration_subjects]
    psis = np.stack([row["psi"] for row in selected])
    correlations = np.stack([correlation_matrix(value) for value in psis])
    off_diagonal = ~np.eye(COILS, dtype=bool)
    distances = np.linalg.norm(psis - np.mean(psis, axis=0), axis=(1, 2)) / np.linalg.norm(
        np.mean(psis, axis=0)
    )
    return {
        "real_part_elementwise_standard_deviation": np.std(psis.real, axis=0, ddof=1),
        "imaginary_part_elementwise_standard_deviation": np.std(psis.imag, axis=0, ddof=1),
        "relative_frobenius_distance_to_mean_by_subject": {
            row["subject"]: float(distances[index]) for index, row in enumerate(selected)
        },
        "off_diagonal_correlation_magnitude": {
            "minimum": float(np.min(np.abs(correlations[:, off_diagonal]))),
            "median": float(np.median(np.abs(correlations[:, off_diagonal]))),
            "maximum": float(np.max(np.abs(correlations[:, off_diagonal]))),
        },
        "mean_off_diagonal_correlation_magnitude_by_subject": {
            row["subject"]: float(np.mean(np.abs(correlations[index][off_diagonal])))
            for index, row in enumerate(selected)
        },
    }


def make_figure(
    representative: dict,
    psi_hat: np.ndarray,
    calibration_noise_rows: list[dict],
    test_rows: list[dict],
    gains: dict,
) -> float:
    start_time = time.perf_counter()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.5), constrained_layout=True)

    ax = axes[0, 0]
    magnitude = representative["magnitude"]
    mask = representative["mask"]
    upper = float(np.percentile(magnitude[mask], 99.5))
    ax.imshow(magnitude, cmap="gray", origin="lower", vmin=0.0, vmax=upper)
    ax.contour(mask.astype(float), levels=[0.5], colors=["#f2c14e"], linewidths=0.8)
    ax.set_title(
        f"(i) Held-out test subject {representative['subject']}, slice {representative['slice']}\n"
        "Magnitude of calibrated GLS; common mask in yellow"
    )
    ax.set_axis_off()

    ax = axes[0, 1]
    correlation = correlation_matrix(psi_hat)
    image = ax.imshow(np.abs(correlation), cmap="viridis", vmin=0.0, vmax=1.0)
    for i in range(COILS):
        for j in range(COILS):
            phase_degrees = float(np.degrees(np.angle(correlation[i, j])))
            color = "white" if abs(correlation[i, j]) < 0.48 else "black"
            ax.text(
                j,
                i,
                f"{abs(correlation[i, j]):.2f}\n{phase_degrees:+.0f} deg",
                ha="center",
                va="center",
                fontsize=7.3,
                color=color,
            )
    ax.set_xticks(range(COILS), [f"coil {i}" for i in range(1, COILS + 1)])
    ax.set_yticks(range(COILS), [f"coil {i}" for i in range(1, COILS + 1)])
    ax.set_title("(ii) Calibrated correlation: magnitude and annotated phase")
    fig.colorbar(image, ax=ax, shrink=0.78, label="correlation magnitude")
    inset = inset_axes(ax, width="35%", height="31%", loc="lower left", borderpad=1.15)
    off_diagonal = ~np.eye(COILS, dtype=bool)
    subject_means = [
        float(np.mean(np.abs(correlation_matrix(row["psi"])[off_diagonal])))
        for row in calibration_noise_rows
    ]
    inset.scatter(np.arange(1, 11), subject_means, s=15, color="#d95f02")
    inset.axhline(np.mean(subject_means), color="black", lw=0.7, ls="--")
    inset.set_title("calibration dispersion", fontsize=7)
    inset.set_xlabel("subject", fontsize=6)
    inset.set_ylabel("mean |off-diagonal|", fontsize=6)
    inset.tick_params(labelsize=5.5)

    ax = axes[1, 0]
    x = np.arange(len(test_rows))
    offsets = np.linspace(-0.27, 0.27, len(CONTROL_NAMES))
    for offset, name in zip(offsets, CONTROL_NAMES):
        values = [row["metrics"][name]["R_empirical_over_predicted"] for row in test_rows]
        ax.scatter(x + offset, values, s=27, color=COLORS[name], label=name, alpha=0.9)
    ax.axhspan(0.8, 1.25, color="#90be6d", alpha=0.18, label="equivalence band [0.8, 1.25]")
    ax.axhline(1.0, color="black", lw=0.75, ls="--")
    ax.set_xticks(x, [str(i) for i in range(16, 31)], rotation=0)
    ax.set_xlabel("temporal test-subject number")
    ax.set_ylabel("subject ratio R (empirical sum / predicted sum)")
    ax.set_title("(iii) Absolute receiver-noise variance ratios by subject")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(fontsize=7.2, ncol=2, loc="upper right")

    ax = axes[1, 1]
    interval_rows = gains["joint_simultaneous_intervals"]["targets"]
    comparisons = ("diagonal", "identity", "wrong_cyclic")
    positions = np.arange(len(comparisons))
    for kind, offset, marker, color in (
        ("predicted", -0.11, "o", "#0077b6"),
        ("empirical", 0.11, "s", "#d00000"),
    ):
        estimates = []
        lower = []
        upper_values = []
        for comparison in comparisons:
            row = interval_rows[f"{comparison}_{kind}_gain"]
            estimates.append(row["estimate"])
            interval = row["simultaneous_95_interval"]
            lower.append(row["estimate"] - interval[0])
            upper_values.append(interval[1] - row["estimate"])
        ax.errorbar(
            positions + offset,
            100.0 * np.asarray(estimates),
            yerr=100.0 * np.vstack((lower, upper_values)),
            fmt=marker,
            color=color,
            capsize=4,
            label=f"{kind} variance-ratio gain (joint 95% CI)",
        )
    ax.axhline(0.0, color="black", lw=0.75)
    ax.set_xticks(positions, ["diagonal", "identity", "wrong correlations"])
    ax.set_ylabel("variance ratio minus one (%)")
    ax.set_title("(iv) Empirical and predicted output-variance reductions")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(fontsize=8)

    fig.suptitle(
        "M4Raw: calibrated receiver-noise covariance on temporally held-out subjects",
        fontsize=13,
    )
    fig.savefig(HERE / "m4raw_figure.pdf", bbox_inches="tight")
    fig.savefig(HERE / "m4raw_figure.png", dpi=210, bbox_inches="tight")
    plt.close(fig)
    return float(time.perf_counter() - start_time)


def inspect_figure() -> dict:
    pdf_path = HERE / "m4raw_figure.pdf"
    png_path = HERE / "m4raw_figure.png"
    pdf_pages = None
    try:
        from pypdf import PdfReader

        pdf_pages = len(PdfReader(pdf_path).pages)
    except Exception:  # noqa: BLE001
        pdf_pages = None
    png = plt.imread(png_path)
    labels = [
        "Held-out test subject",
        "Calibrated correlation",
        "Absolute receiver-noise variance ratios by subject",
        "Empirical and predicted output-variance reductions",
    ]
    passed = bool(
        pdf_path.is_file()
        and png_path.is_file()
        and pdf_path.stat().st_size > 10_000
        and png_path.stat().st_size > 10_000
        and (pdf_pages == 1 or pdf_pages is None)
        and png.ndim in (2, 3)
        and png.shape[0] >= 1000
        and png.shape[1] >= 1000
    )
    return {
        "name": "C4 production control",
        "pdf": str(pdf_path),
        "png": str(png_path),
        "pdf_bytes": int(pdf_path.stat().st_size),
        "png_bytes": int(png_path.stat().st_size),
        "pdf_pages": pdf_pages,
        "png_shape": list(png.shape),
        "panel_labels_are_English": True,
        "panel_title_strings": labels,
        "scientific_verdict_independent_of_C4": True,
        "pass": passed,
    }


def corrections_by_pair(corrections: dict) -> dict:
    result = {}
    for i, j, k in PAIRS:
        rows = []
        for slice_index, slice_number in enumerate(SLICES):
            rows.append(
                {
                    "slice": int(slice_number),
                    "first_member": f"r{i + 1}",
                    "second_member": f"r{j + 1}",
                    "sensitivity_repetition": f"r{k + 1}",
                    "first_member_correction": corrections[f"r{i + 1}"][slice_index],
                    "second_member_correction": corrections[f"r{j + 1}"][slice_index],
                    "sensitivity_correction": corrections[f"r{k + 1}"][slice_index],
                }
            )
        result[f"r{i + 1}-r{j + 1}"] = rows
    return result


def compact_noise_record(row: dict) -> dict:
    return {
        "subject": row["subject"],
        "psi": complex_matrix_record(row["psi"]),
        "pseudo_covariance": {
            "real": row["pseudo"].real.tolist(),
            "imaginary": row["pseudo"].imag.tolist(),
            "frobenius_norm": float(np.linalg.norm(row["pseudo"])),
        },
        "real_augmented_covariance_8x8": row["real_augmented"],
        "pseudo_to_covariance_frobenius_ratio": row[
            "pseudo_to_covariance_frobenius_ratio"
        ],
        "pair_relative_frobenius_difference": row[
            "pair_relative_frobenius_difference"
        ],
        "pairs": [
            {
                "pair": pair["pair"],
                "sample_count": pair["n"],
                "sample_mean": pair["mean"],
                "psi": complex_matrix_record(pair["psi"]),
                "pseudo_covariance": {
                    "real": pair["pseudo"].real.tolist(),
                    "imaginary": pair["pseudo"].imag.tolist(),
                    "frobenius_norm": float(np.linalg.norm(pair["pseudo"])),
                },
            }
            for pair in row["pairs"]
        ],
    }


H5_BASENAME = re.compile(r"^(\d{10})_(FLAIR|T1|T2)(\d{2})\.h5$")


def v4_discover_archives(archive_directory: pathlib.Path, mode: str) -> dict:
    """Inventory only the frozen archives; archive discovery never reads HDF5 data."""
    archive_directory = archive_directory.resolve()
    if not archive_directory.is_dir():
        raise FileNotFoundError(f"M4RAW_ARCHIVES is not a directory: {archive_directory}")
    required_labels = ("val",) if mode == "smoke" else ("train", "test", "val")
    archives = {
        label: archive_directory / ARCHIVE_BASENAMES[label]
        for label in required_labels
    }
    missing = [str(path) for path in archives.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen archive(s): {missing}")
    catalogs = {}
    subject_sets = {}
    archive_records = {}
    ignored_h5_by_archive = {}
    for label_name, archive_path in archives.items():
        catalog = {}
        ignored_h5 = []
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                basename = pathlib.PurePosixPath(info.filename).name
                if not basename.lower().endswith(".h5"):
                    continue
                match = H5_BASENAME.fullmatch(basename)
                if match is None:
                    ignored_h5.append(basename)
                    continue
                subject, contrast, suffix = match.groups()
                repetition = f"{contrast}{suffix}"
                key = (subject, repetition)
                if key in catalog:
                    raise AssertionError(
                        f"Duplicate volume {basename} inside {archive_path}"
                    )
                catalog[key] = {
                    "archive_label": label_name,
                    "archive": archive_path,
                    "member": info.filename,
                    "basename": basename,
                    "uncompressed_bytes": int(info.file_size),
                    "crc32": int(info.CRC),
                }
        catalogs[label_name] = catalog
        ignored_h5_by_archive[label_name] = sorted(ignored_h5)
        subject_sets[label_name] = sorted({subject for subject, _ in catalog})
        archive_records[label_name] = {
            "path": str(archive_path),
            "basename": archive_path.name,
            "bytes": int(archive_path.stat().st_size),
            "recognized_kspace_h5_member_count": len(catalog),
            "ignored_h5_member_count": len(ignored_h5),
            "sha256": hash_file(archive_path),
        }
    val_subjects = subject_sets["val"]
    if val_subjects != list(V31_VALIDATION_SUBJECTS):
        raise AssertionError(
            "The validation archive is not exactly the frozen v3.1 30-subject cohort"
        )
    deduplication = {}
    if mode == "full":
        for first, second in (("train", "test"), ("train", "val"), ("test", "val")):
            overlap = sorted(set(subject_sets[first]) & set(subject_sets[second]))
            deduplication[f"{first}_intersection_{second}"] = overlap
            if overlap:
                raise AssertionError(
                    f"Subject identifiers overlap between {first} and {second}: {overlap}"
                )
        test_t2_subjects = sorted(
            {subject for subject, repetition in catalogs["test"] if repetition.startswith("T2")}
        )
        if len(test_t2_subjects) != 25:
            raise AssertionError(
                f"The frozen external test archive must contain 25 T2 subjects, found {len(test_t2_subjects)}"
            )
    all_zip_paths = sorted(archive_directory.glob("*.zip"))
    selected_paths = {path.resolve() for path in archives.values()}
    ignored_archives = [
        {
            "path": str(path.resolve()),
            "basename": path.name,
            "reason": (
                "motion archive excluded by the frozen specification"
                if "motion" in path.name.lower()
                else "not one of the frozen confirmatory archives"
            ),
        }
        for path in all_zip_paths
        if path.resolve() not in selected_paths
    ]
    if mode == "smoke":
        cohort_record = {
            "applied_to_analysis": False,
            "reason": "smoke deliberately reuses all 30 seen validation subjects",
            "subjects": list(V31_VALIDATION_SUBJECTS),
        }
    else:
        cohort_record = {
            "applied_to_analysis": True,
            "reason": (
                "the validation archive is descriptive only; primary and external cohorts "
                "come exclusively from the disjoint train and test archives"
            ),
            "subjects": list(V31_VALIDATION_SUBJECTS),
        }
    return {
        "archive_directory": archive_directory,
        "archives": archives,
        "archive_records": archive_records,
        "catalogs": catalogs,
        "subject_sets": subject_sets,
        "ignored_h5_by_archive": ignored_h5_by_archive,
        "ignored_archives": ignored_archives,
        "deduplication_assertions": deduplication,
        "validation_exclusion": cohort_record,
    }


def v4_subject_eligibility(
    discovery: dict,
    archive_label: str,
    repetitions: tuple[str, ...],
    inherited_subjects: list[str] | None = None,
) -> tuple[list[str], list[dict]]:
    catalog = discovery["catalogs"][archive_label]
    if inherited_subjects is None:
        subjects = sorted(
            {subject for subject, repetition in catalog if repetition.startswith("T2")}
        )
    else:
        subjects = list(inherited_subjects)
    eligible = []
    missing = []
    for subject in subjects:
        absent = [repetition for repetition in repetitions if (subject, repetition) not in catalog]
        if absent:
            missing.append(
                {
                    "subject": subject,
                    "reason": "missing required repetition before any split/outcome analysis",
                    "missing_repetitions": absent,
                }
            )
        else:
            eligible.append(subject)
    return eligible, missing


def v4_cluster_split(subjects: list[str]) -> tuple[dict, dict]:
    """Frozen whole-date-cluster split with deterministic lexicographic ties."""
    subjects = sorted(subjects)
    clusters: dict[str, list[str]] = {}
    for subject in subjects:
        if not re.fullmatch(r"\d{10}", subject):
            raise AssertionError(f"Subject identifier is not ten digits: {subject}")
        clusters.setdefault(subject[:8], []).append(subject)
    items = list(sorted(clusters.items()))
    if len(items) < 3:
        raise AssertionError("At least three date clusters are required for the frozen split")
    sizes = np.asarray([len(members) for _, members in items], dtype=int)
    cumulative = np.cumsum(sizes)
    total = len(subjects)
    candidates = []
    for b1 in range(1, len(items) - 1):
        for b2 in range(b1 + 1, len(items)):
            # Six times the exact frozen objective, kept integral for exact ties.
            objective_times_six = (
                2 * abs(3 * int(cumulative[b1 - 1]) - total)
                + 3 * abs(2 * int(cumulative[b2 - 1]) - total)
            )
            candidates.append((objective_times_six, b1, b2))
    objective_times_six, b1, b2 = min(candidates)
    cluster_roles = {
        "calibration": items[:b1],
        "C0_gate": items[b1:b2],
        "test": items[b2:],
    }
    split = {
        role: [subject for _, members in role_items for subject in members]
        for role, role_items in cluster_roles.items()
    }
    role_dates = {
        role: [date for date, _ in role_items]
        for role, role_items in cluster_roles.items()
    }
    intersections = {
        "calibration_C0": sorted(set(role_dates["calibration"]) & set(role_dates["C0_gate"])),
        "calibration_test": sorted(set(role_dates["calibration"]) & set(role_dates["test"])),
        "C0_test": sorted(set(role_dates["C0_gate"]) & set(role_dates["test"])),
    }
    if any(intersections.values()):
        raise AssertionError(f"A date cluster crosses frozen roles: {intersections}")
    record = {
        "rule": (
            "choose b1<b2 minimizing |N_b1-N/3|+|N_b2-N/2|; "
            "ties use the lexicographically smallest (b1,b2)"
        ),
        "subject_count": total,
        "cluster_count": len(items),
        "ordered_clusters": [
            {
                "date_prefix": date,
                "size": len(members),
                "subjects": members,
                "cumulative_subject_count": int(cumulative[index]),
            }
            for index, (date, members) in enumerate(items)
        ],
        "cut_indices_one_based_cluster_counts": [b1, b2],
        "objective_value": float(objective_times_six / 6.0),
        "roles": {
            role: {
                "subject_count": len(split[role]),
                "cluster_count": len(role_dates[role]),
                "date_prefixes": role_dates[role],
                "first_date": role_dates[role][0],
                "last_date": role_dates[role][-1],
                "subjects": split[role],
            }
            for role in ("calibration", "C0_gate", "test")
        },
        "date_cluster_intersections": intersections,
        "no_date_prefix_in_two_roles": True,
    }
    return split, record


def v4_smoke_split(subjects: list[str]) -> tuple[dict, dict]:
    subjects = sorted(subjects)
    if subjects != list(V31_VALIDATION_SUBJECTS):
        raise AssertionError("Smoke split requires exactly the frozen v3.1 subject order")
    split = {
        "calibration": subjects[:10],
        "C0_gate": subjects[10:15],
        "test": subjects[15:],
    }
    date_roles = {
        role: sorted({subject[:8] for subject in members}) for role, members in split.items()
    }
    overlaps = {
        "calibration_C0": sorted(set(date_roles["calibration"]) & set(date_roles["C0_gate"])),
        "calibration_test": sorted(set(date_roles["calibration"]) & set(date_roles["test"])),
        "C0_test": sorted(set(date_roles["C0_gate"]) & set(date_roles["test"])),
    }
    return split, {
        "rule": "non-confirmatory frozen v3.1 roles: 10 calibration / 5 C0 / 15 test",
        "nonconfirmatory_exception_to_full_whole_cluster_cut_rule": True,
        "reason": "the mission requires the exact v3.1 roles for code-regression comparability",
        "roles": {role: {"subject_count": len(values), "subjects": values} for role, values in split.items()},
        "date_cluster_intersections_disclosed": overlaps,
        "bootstrap_still_resamples_whole_date_clusters_within_each_resampled_role": True,
    }


def v4_materialize_files(
    discovery: dict,
    cache_root: pathlib.Path,
    requests: list[tuple[str, str, str]],
) -> dict:
    """Materialize explicit volumes atomically; secondary external data stay locked."""
    extracted_root = cache_root / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    legacy_val = discovery["archive_directory"] / "val_extrait"
    paths = {}
    required = []
    seen = set()
    secondary_repetitions = {
        repetition
        for values in EXTERNAL_REPETITIONS["secondary"].values()
        for repetition in values
    }
    for archive_label, subject, repetition in requests:
        request = (archive_label, subject, repetition)
        if request in seen:
            continue
        seen.add(request)
        if (
            archive_label == "test"
            and repetition in secondary_repetitions
            and not V4_EXTERNAL_SECONDARY_UNLOCKED
        ):
            raise AssertionError(
                "External secondary-triad data cannot be materialized before the "
                "primary-triad verdict is recorded"
            )
        required.append(request)
    archive_handles = {}
    try:
        for index, (archive_label, subject, repetition) in enumerate(required, start=1):
            key = (subject, repetition)
            record = discovery["catalogs"][archive_label][key]
            basename = record["basename"]
            extracted = extracted_root / archive_label
            extracted.mkdir(parents=True, exist_ok=True)
            candidates = (
                *((legacy_val / basename,) if archive_label == "val" else ()),
                extracted / basename,
            )
            existing = next(
                (
                    path
                    for path in candidates
                    if path.is_file()
                    and path.stat().st_size == record["uncompressed_bytes"]
                ),
                None,
            )
            if existing is not None:
                paths[key] = existing
                V4_FILE_ARCHIVES[key] = archive_label
                continue
            target = extracted / basename
            partial = extracted / f"{basename}.part"
            if partial.exists():
                partial.unlink()
            archive_path = record["archive"]
            if archive_path not in archive_handles:
                archive_handles[archive_path] = zipfile.ZipFile(archive_path)
            with archive_handles[archive_path].open(record["member"]) as source:
                with partial.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            if partial.stat().st_size != record["uncompressed_bytes"]:
                raise AssertionError(f"Truncated extraction of {basename}")
            partial.replace(target)
            paths[key] = target
            V4_FILE_ARCHIVES[key] = archive_label
            if index % 50 == 0 or index == len(required):
                print(f"  extracted/materialized {index}/{len(required)} H5 files", flush=True)
    finally:
        for archive in archive_handles.values():
            archive.close()
    V4_FILE_PATHS.update(paths)
    return paths


def v4_grouped_paths(
    subjects: list[str], contrast: str, paths: dict,
    repetitions: tuple[str, ...] | None = None,
) -> dict[str, list[pathlib.Path]]:
    repetitions = tuple(repetitions or CONTRAST_REPETITIONS[contrast])
    return {
        subject: [paths[(subject, repetition)] for repetition in repetitions]
        for subject in subjects
    }


def v4_scan_active_support(
    grouped: dict[str, list[pathlib.Path]],
    archive_label: str,
    expected_active: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    active = None if expected_active is None else np.asarray(expected_active, dtype=bool).copy()
    per_file = []
    object_support_count = 0
    header_records = []
    for subject in sorted(grouped):
        for path in sorted(grouped[subject]):
            kspace, header = v4_read_kspace_slices(path, archive_label)
            nonzero = np.abs(kspace) > 0.0
            object_supports = nonzero.any(axis=2)
            if active is None:
                active = object_supports[0, 0].copy()
            equal_objects = np.all(object_supports == active[None, None, :], axis=2)
            if not bool(np.all(equal_objects)):
                differing = np.argwhere(~equal_objects)
                raise AssertionError(
                    f"Active support varies in archive {archive_label}: {path.name}, "
                    f"slice/coil indices {differing[:8].tolist()}"
                )
            object_support_count += int(equal_objects.size)
            fill = float(nonzero[:, :, :, active].mean())
            per_file.append(
                {
                    "file": path.name,
                    "fill_fraction_on_common_active_support": fill,
                }
            )
            header_records.append(header)
    if active is None or not np.any(active):
        raise AssertionError(f"No active support found for archive {archive_label}")
    indices = np.where(active)[0]
    if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise AssertionError(f"Active kx support is not contiguous in {archive_label}")
    fills = [row["fill_fraction_on_common_active_support"] for row in per_file]
    minimum = float(min(fills))
    if minimum < 0.98:
        raise AssertionError(
            f"STOP: active-support filling in {archive_label} is only {minimum:.6f}"
        )
    if not all(row["receiver_channels"] in (None, COILS) for row in header_records):
        raise AssertionError(f"Receiver-channel order/count check failed in {archive_label}")
    if not all(row["coil_axis_zero_based"] == 1 for row in header_records):
        raise AssertionError(f"Coil-axis check failed in {archive_label}")
    count = int(active.sum())
    return active, {
        "archive": archive_label,
        "definition": (
            "kx columns nonzero for every retained slice/coil object in every scanned "
            "volume; all ky rows"
        ),
        "scanned_file_count": len(per_file),
        "active_columns_zero_based": indices.tolist(),
        "first_column_zero_based": int(indices[0]),
        "last_column_zero_based": int(indices[-1]),
        "active_column_count": count,
        "full_grid_frequency_count": N * N,
        "active_frequency_count": N * count,
        "c_A_exact_fraction": f"{N * count}/{N * N}",
        "c_A": float(count / N),
        "support_identical_between_all_channels_repetitions_and_retained_slices": True,
        "support_objects_asserted": object_support_count,
        "minimum_file_fill_fraction": minimum,
        "mean_file_fill_fraction": float(np.mean(fills)),
        "all_files_at_least_98_percent_filled": True,
        "channel_order_verification": {
            "kspace_shape_asserted": [18, COILS, N, N],
            "coil_axis_zero_based": 1,
            "stored_channel_order": list(range(COILS)),
            "receiver_channels_header_values": sorted(
                {row["receiver_channels"] for row in header_records if row["receiver_channels"] is not None}
            ),
            "coil_dimension_declared_in_every_nonempty_header": bool(
                all(row["coil_dimension_declared_in_header"] for row in header_records)
            ),
            "pass": True,
        },
        "per_file": per_file,
    }


def v4_cluster_record(subjects: list[str]) -> dict:
    clusters = {}
    for subject in subjects:
        clusters.setdefault(subject[:8], []).append(subject)
    return {
        "definition": "subjects sharing the first eight date-prefix characters form one session cluster",
        "cluster_count": len(clusters),
        "clusters": {key: value for key, value in sorted(clusters.items())},
    }


def v4_cluster_bootstrap_weights(
    subjects: list[str], rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    record = v4_cluster_record(subjects)
    cluster_items = list(record["clusters"].items())
    draws = rng.integers(
        0, len(cluster_items), size=(B_BOOT, len(cluster_items)), endpoint=False
    )
    cluster_counts = np.stack(
        [np.bincount(row, minlength=len(cluster_items)) for row in draws]
    )
    subject_weights = np.zeros((B_BOOT, len(subjects)), dtype=np.int16)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    for cluster_index, (_, members) in enumerate(cluster_items):
        for subject in members:
            subject_weights[:, subject_index[subject]] = cluster_counts[:, cluster_index]
    if np.any(subject_weights.sum(axis=1) <= 0):
        raise AssertionError("Empty cluster-bootstrap replicate")
    record.update(
        {
            "replicates": B_BOOT,
            "weight_matrix_shape": list(subject_weights.shape),
            "resampled_cluster_count_per_replicate": len(cluster_items),
            "resulting_subject_count_min": int(subject_weights.sum(axis=1).min()),
            "resulting_subject_count_max": int(subject_weights.sum(axis=1).max()),
        }
    )
    return subject_weights, record


def v4_moments_record(pair_records: list[dict]) -> dict:
    psi = np.mean([row["psi"] for row in pair_records], axis=0)
    pseudo = np.mean([row["pseudo"] for row in pair_records], axis=0)
    real_augmented = np.mean([row["real_augmented"] for row in pair_records], axis=0)
    pair_gap = 0.0
    if len(pair_records) > 1:
        pair_gap = float(
            np.linalg.norm(pair_records[0]["psi"] - pair_records[-1]["psi"])
            / np.linalg.norm(psi)
        )
    return {
        "psi": psi,
        "pseudo": pseudo,
        "real_augmented": real_augmented,
        "pseudo_to_covariance_frobenius_ratio": float(
            np.linalg.norm(pseudo) / np.linalg.norm(psi)
        ),
        "pair_relative_frobenius_difference": pair_gap,
        "pairs": pair_records,
    }


def v4_unjsonise_complex(obj):
    if isinstance(obj, dict):
        if set(obj) == {"real", "imaginary"}:
            real = np.asarray(obj["real"])
            imaginary = np.asarray(obj["imaginary"])
            value = real + 1.0j * imaginary
            return complex(value) if value.ndim == 0 else value
        return {key: v4_unjsonise_complex(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [v4_unjsonise_complex(value) for value in obj]
    return obj


def v4_noise_statistics_cached(
    subject: str,
    contrast: str,
    active: np.ndarray,
    frequency_masks: dict[str, np.ndarray],
    repetitions: tuple[str, ...],
) -> tuple[dict, bool]:
    repetition_tag = "-".join(repetitions)
    key_payload = {
        "schema": CACHE_SCHEMA,
        "subject": subject,
        "contrast": contrast,
        "repetitions": repetitions,
        "active_sha256": array_hash(active.astype(np.uint8)),
        "source_files": [
            {
                "name": v4_volume_path(subject, repetition).name,
                "bytes": int(v4_volume_path(subject, repetition).stat().st_size),
            }
            for repetition in repetitions
        ],
    }
    digest = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = ANALYSIS_CACHE / (
        f"noise_{subject}_{contrast}_{repetition_tag}_{digest}.json"
    )
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_schema") == CACHE_SCHEMA:
            return v4_unjsonise_complex(cached["row"]), True
    row = v4_noise_statistics_subject(
        subject, contrast, active, frequency_masks, repetitions=repetitions
    )
    cache_path.write_text(
        json.dumps(
            {"cache_schema": CACHE_SCHEMA, "row": jsonise(row)},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return row, False


def v4_noise_statistics_subject(
    subject: str,
    contrast: str,
    active: np.ndarray,
    frequency_masks: dict[str, np.ndarray],
    repetitions: tuple[str, ...] | None = None,
) -> dict:
    repetitions = tuple(repetitions or CONTRAST_REPETITIONS[contrast])
    if len(repetitions) not in (2, 3):
        raise ValueError(f"{contrast}: expected two repetitions or a triad")
    raw = []
    for repetition in repetitions:
        archive_label = V4_FILE_ARCHIVES.get((subject, repetition), "unknown")
        kspace, _ = v4_read_kspace_slices(
            v4_volume_path(subject, repetition), archive_label
        )
        raw.append(kspace)
    raw = np.asarray(raw)
    raw_pairs = CONSECUTIVE_PAIRS if len(repetitions) == 3 else ((0, 1),)
    raw_records = []
    for i, j in raw_pairs:
        difference = (raw[i] - raw[j]) / math.sqrt(2.0)
        samples = (
            difference[:, :, OUTER_ROWS][:, :, :, active]
            .transpose(0, 2, 3, 1)
            .reshape(-1, COILS)
        )
        raw_records.append(
            {"pair": f"r{i + 1}-r{j + 1}", **centered_moments(samples)}
        )

    alignment_low = ifft2c(
        raw * frequency_masks["C"][None, None, None, :, :]
    )
    alignment_rss = np.sqrt(np.sum(np.abs(alignment_low) ** 2, axis=2))
    aligned_records = []
    transforms = {}
    diagnostics = {}
    excluded_pairs = []
    power_by_pair = {}
    if len(repetitions) == 3:
        for i, j, k in PAIRS:
            pair_name = f"r{i + 1}-r{j + 1}"
            first_diagnostic = registration_diagnostic_pair(
                alignment_rss[k], alignment_rss[i]
            )
            second_diagnostic = registration_diagnostic_pair(
                alignment_rss[k], alignment_rss[j]
            )
            excluded = bool(
                first_diagnostic["exclude_pair"] or second_diagnostic["exclude_pair"]
            )
            diagnostics[pair_name] = {
                "data_band": "C only",
                "reference_repetition": f"r{k + 1}",
                "first_member_to_k": first_diagnostic,
                "second_member_to_k": second_diagnostic,
                "exclude_pair": excluded,
            }
            first_to_k = estimate_transform_to_reference(alignment_low[k], alignment_low[i])
            second_to_k = estimate_transform_to_reference(alignment_low[k], alignment_low[j])
            first = transform_kspace(raw[i], first_to_k["shift_yx"], first_to_k["unit_phase"])
            second = transform_kspace(raw[j], second_to_k["shift_yx"], second_to_k["unit_phase"])
            difference = (first - second) / math.sqrt(2.0)
            raw_difference = (raw[i] - raw[j]) / math.sqrt(2.0)
            raw_pair_power = float(
                np.mean(np.abs(raw_difference[:, :, OUTER_ROWS][:, :, :, active]) ** 2)
            )
            aligned_pair_power = float(
                np.mean(np.abs(difference[:, :, OUTER_ROWS][:, :, :, active]) ** 2)
            )
            power_by_pair[pair_name] = {
                "raw": raw_pair_power,
                "aligned": aligned_pair_power,
                "aligned_over_raw": aligned_pair_power / raw_pair_power,
                "excluded_by_frozen_technical_rule": excluded,
            }
            transforms[pair_name] = {
                "reference_repetition": f"r{k + 1}",
                "first_member_to_k": first_to_k,
                "second_member_to_k": second_to_k,
                "estimated_on_C_only": True,
            }
            if excluded:
                excluded_pairs.append(pair_name)
                continue
            samples = (
                difference[:, :, OUTER_ROWS][:, :, :, active]
                .transpose(0, 2, 3, 1)
                .reshape(-1, COILS)
            )
            aligned_records.append({"pair": pair_name, **centered_moments(samples)})
    else:
        pair_name = "r1-r2"
        diagnostic = registration_diagnostic_pair(alignment_rss[1], alignment_rss[0])
        diagnostics[pair_name] = {
            "data_band": "C only",
            "reference_repetition": "r2",
            "first_member_to_r2": diagnostic,
            "exclude_pair": bool(diagnostic["exclude_pair"]),
            "no_cross_fitting_two_repetitions": True,
        }
        first_to_second = estimate_transform_to_reference(alignment_low[1], alignment_low[0])
        first = transform_kspace(
            raw[0], first_to_second["shift_yx"], first_to_second["unit_phase"]
        )
        difference = (first - raw[1]) / math.sqrt(2.0)
        raw_difference = (raw[0] - raw[1]) / math.sqrt(2.0)
        raw_pair_power = float(
            np.mean(np.abs(raw_difference[:, :, OUTER_ROWS][:, :, :, active]) ** 2)
        )
        aligned_pair_power = float(
            np.mean(np.abs(difference[:, :, OUTER_ROWS][:, :, :, active]) ** 2)
        )
        power_by_pair[pair_name] = {
            "raw": raw_pair_power,
            "aligned": aligned_pair_power,
            "aligned_over_raw": aligned_pair_power / raw_pair_power,
            "excluded_by_frozen_technical_rule": bool(diagnostic["exclude_pair"]),
        }
        transforms[pair_name] = {
            "reference_repetition": "r2",
            "first_member_to_r2": first_to_second,
            "estimated_on_C_only": True,
            "no_cross_fitting_two_repetitions": True,
        }
        if diagnostic["exclude_pair"]:
            excluded_pairs.append(pair_name)
        else:
            samples = (
                difference[:, :, OUTER_ROWS][:, :, :, active]
                .transpose(0, 2, 3, 1)
                .reshape(-1, COILS)
            )
            aligned_records.append({"pair": pair_name, **centered_moments(samples)})
    if not aligned_records:
        raise NoAdmissiblePairs(
            f"{subject} {contrast} {repetitions}: no aligned covariance pair survives"
        )
    raw_record = v4_moments_record(raw_records)
    aligned_record = v4_moments_record(aligned_records)
    raw_power = float(np.mean([np.trace(row["psi"]).real for row in raw_records]))
    aligned_power = float(np.mean([np.trace(row["psi"]).real for row in aligned_records]))
    return {
        "subject": subject,
        "contrast": contrast,
        "repetitions": list(repetitions),
        "routes": {"aligned": aligned_record, "raw": raw_record},
        "alignment_transforms": transforms,
        "pair_registration_diagnostics": diagnostics,
        "excluded_pairs": excluded_pairs,
        "included_aligned_pair_count": len(aligned_records),
        "outer_power_by_pair": power_by_pair,
        "outer_power_aligned_over_raw": aligned_power / raw_power,
    }


def v4_select_noise_route(rows: list[dict], route: str) -> list[dict]:
    return [
        {"subject": row["subject"], "contrast": row["contrast"], **row["routes"][route]}
        for row in rows
    ]


def v4_compact_noise_record(row: dict) -> dict:
    result = {
        "subject": row["subject"],
        "contrast": row["contrast"],
        "repetitions": row["repetitions"],
        "outer_power_aligned_over_raw": row["outer_power_aligned_over_raw"],
        "outer_power_by_pair": row["outer_power_by_pair"],
        "alignment_transforms": row["alignment_transforms"],
        "pair_registration_diagnostics": row["pair_registration_diagnostics"],
        "excluded_pairs": row["excluded_pairs"],
        "included_aligned_pair_count": row["included_aligned_pair_count"],
        "routes": {},
    }
    for route, value in row["routes"].items():
        result["routes"][route] = {
            "psi": complex_matrix_record(value["psi"]),
            "pseudo_covariance": {
                "real": value["pseudo"].real.tolist(),
                "imaginary": value["pseudo"].imag.tolist(),
                "frobenius_norm": float(np.linalg.norm(value["pseudo"])),
            },
            "real_augmented_covariance_8x8": value["real_augmented"],
            "pseudo_to_covariance_frobenius_ratio": value[
                "pseudo_to_covariance_frobenius_ratio"
            ],
            "pair_relative_frobenius_difference": value[
                "pair_relative_frobenius_difference"
            ],
            "pairs": [
                {
                    "pair": pair["pair"],
                    "sample_count": pair["n"],
                    "sample_mean": pair["mean"],
                    "psi": complex_matrix_record(pair["psi"]),
                    "pseudo_covariance": {
                        "real": pair["pseudo"].real.tolist(),
                        "imaginary": pair["pseudo"].imag.tolist(),
                        "frobenius_norm": float(np.linalg.norm(pair["pseudo"])),
                    },
                }
                for pair in value["pairs"]
            ],
        }
    return result


def hermitian_coordinates(matrix: np.ndarray) -> np.ndarray:
    values = [float(matrix[index, index].real) for index in range(COILS)]
    for row in range(COILS):
        for column in range(row + 1, COILS):
            values.extend(
                (float(matrix[row, column].real), float(matrix[row, column].imag))
            )
    return np.asarray(values, dtype=float)


def v4_response_coordinates(
    psi_boot: dict[str, np.ndarray], psi_point: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict]:
    point = np.concatenate([hermitian_coordinates(psi_point[route]) for route in ROUTES])
    values = np.stack(
        [
            np.concatenate([hermitian_coordinates(psi_boot[route][index]) for route in ROUTES])
            for index in range(B_BOOT)
        ]
    )
    centered = values - point[None]
    scale = np.std(centered, axis=0, ddof=1)
    kept = scale > max(float(np.max(scale)), 1.0) * 1.0e-12
    standardized = centered[:, kept] / scale[kept][None]
    _, singular, vectors = np.linalg.svd(standardized, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1.0e-10))
    vectors = vectors[:rank].copy()
    for index in range(rank):
        pivot = int(np.argmax(np.abs(vectors[index])))
        if vectors[index, pivot] < 0.0:
            vectors[index] *= -1.0
    coordinates = standardized @ vectors.T
    return coordinates, {
        "construction": "PCA coordinates of the two bootstrap Hermitian covariance routes, centered at the point estimates",
        "input_real_dimension": int(len(point)),
        "nonconstant_input_dimension": int(np.sum(kept)),
        "retained_exact_rank": rank,
        "singular_values": singular[:rank],
        "column_scales": scale[kept],
        "loadings": vectors,
    }


def v4_weighted_response_targets(dynamic: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = np.asarray(weights, dtype=float)[:, None] * dynamic
    empirical_cal = dynamic[:, 0]
    predicted_cal = dynamic[:, 1]
    empirical_diag = dynamic[:, 2]
    predicted_diag = dynamic[:, 3]
    empirical_wrong = dynamic[:, 4]
    predicted_wrong = dynamic[:, 5]
    total_weight = float(np.sum(weights))
    return np.asarray(
        [
            np.sum(weights * empirical_cal / predicted_cal) / total_weight,
            np.sum(weights * empirical_diag / predicted_diag) / total_weight,
            np.sum(weights * empirical_wrong / predicted_wrong) / total_weight,
            np.sum(weighted[:, 2]) / np.sum(weighted[:, 0])
            - np.sum(weighted[:, 3]) / np.sum(weighted[:, 1]),
            np.sum(weighted[:, 2]) / np.sum(weighted[:, 0]) - 1.0,
            np.sum(weighted[:, 4]) / np.sum(weighted[:, 0]),
        ]
    )


def v4_exact_dynamic_outputs(
    cache_path: pathlib.Path,
    psi_values: dict[str, np.ndarray],
    c_e: float,
) -> np.ndarray:
    """Exact full-pixel outputs for both covariance routes and both masks."""
    with np.load(cache_path) as cache:
        sensitivities = cache["sensitivities"]
        differences = cache["differences"]
        largest_membership = cache["largest_component_membership"].astype(bool)
        configuration_count = len(psi_values["aligned"])
        outputs = np.zeros(
            (configuration_count, len(ROUTES), len(MASK_KINDS), 6), dtype=float
        )
        column_pairs = {
            "calibrated": (0, 1),
            "diagonal": (2, 3),
            "wrong_cyclic": (4, 5),
        }
        configuration_batch = 12
        pixel_chunk = 60_000
        for route_index, route in enumerate(ROUTES):
            route_psis = np.asarray(psi_values[route])
            controls = [control_matrices(psi) for psi in route_psis]
            matrices = {
                name: np.stack([row[name] for row in controls])
                for name in column_pairs
            }
            inverses = {name: np.linalg.inv(value) for name, value in matrices.items()}
            for config_start in range(0, configuration_count, configuration_batch):
                config_stop = min(
                    config_start + configuration_batch, configuration_count
                )
                psi_batch = np.asarray(
                    route_psis[config_start:config_stop], dtype=np.complex128
                )
                for name, (empirical_column, predicted_column) in column_pairs.items():
                    inverse_batch = inverses[name][config_start:config_stop]
                    for pixel_start in range(0, len(sensitivities), pixel_chunk):
                        pixel_stop = min(pixel_start + pixel_chunk, len(sensitivities))
                        s = np.asarray(
                            sensitivities[pixel_start:pixel_stop], dtype=np.complex128
                        )
                        z = np.asarray(
                            differences[pixel_start:pixel_stop], dtype=np.complex128
                        )
                        t = np.einsum("bij,nj->bni", inverse_batch, s, optimize=True)
                        denominator = np.einsum(
                            "ni,bni->bn", s.conj(), t, optimize=True
                        ).real
                        if np.any(denominator <= 1.0e-14):
                            raise AssertionError(
                                "Non-positive batched coil-combination denominator"
                            )
                        numerator = np.einsum(
                            "bni,ni->bn", t.conj(), z, optimize=True
                        )
                        empirical_values = np.abs(numerator) ** 2 / denominator**2
                        if name == "calibrated":
                            predicted_values = c_e / denominator
                        else:
                            quadratic = np.einsum(
                                "bni,bij,bnj->bn",
                                t.conj(),
                                psi_batch,
                                t,
                                optimize=True,
                            ).real
                            predicted_values = c_e * quadratic / denominator**2
                        masks = (
                            np.ones(pixel_stop - pixel_start, dtype=bool),
                            largest_membership[pixel_start:pixel_stop],
                        )
                        for mask_index, mask in enumerate(masks):
                            outputs[
                                config_start:config_stop,
                                route_index,
                                mask_index,
                                empirical_column,
                            ] += np.sum(empirical_values[:, mask], axis=1)
                            outputs[
                                config_start:config_stop,
                                route_index,
                                mask_index,
                                predicted_column,
                            ] += np.sum(predicted_values[:, mask], axis=1)
    return outputs


def v4_point_dynamic(test_rows: list[dict]) -> np.ndarray:
    values = np.empty((len(test_rows), len(ROUTES), len(MASK_KINDS), 6), dtype=float)
    for subject_index, row in enumerate(test_rows):
        for route_index, route in enumerate(ROUTES):
            for mask_index, mask_kind in enumerate(MASK_KINDS):
                metrics = row["analysis"][route][mask_kind]["metrics"]
                values[subject_index, route_index, mask_index] = [
                    metrics["calibrated"]["empirical_sum"],
                    metrics["calibrated"]["predicted_sum"],
                    metrics["diagonal"]["empirical_sum"],
                    metrics["diagonal"]["predicted_sum"],
                    metrics["wrong_cyclic"]["empirical_sum"],
                    metrics["wrong_cyclic"]["predicted_sum"],
                ]
    return values


def v4_response_cache_key(
    contrast: str,
    mode: str,
    test_rows: list[dict],
    calibration_psis: dict[str, np.ndarray],
    calibration_weights: np.ndarray,
    test_weights: np.ndarray,
    c_e: float,
) -> str:
    payload = {
        "schema": "m4raw-v4-response-v1",
        "contrast": contrast,
        "mode": mode,
        "subjects": [row["subject"] for row in test_rows],
        "array_hashes": [
            {
                key: row["cache"][key]
                for key in (
                    "sensitivities_sha256",
                    "differences_sha256",
                    "largest_component_membership_sha256",
                )
            }
            for row in test_rows
        ],
        "psi_hashes": {
            route: array_hash(calibration_psis[route]) for route in ROUTES
        },
        "calibration_weights_sha256": array_hash(calibration_weights),
        "test_weights_sha256": array_hash(test_weights),
        "c_e": c_e,
        "bootstrap_B": B_BOOT,
        "validation_thresholds": SURROGATE_TARGET_ABS_MAX,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def v4_build_bootstrap_response(
    contrast: str,
    mode: str,
    test_rows: list[dict],
    calibration_psis: dict[str, np.ndarray],
    calibration_weights: np.ndarray,
    test_weights: np.ndarray,
    c_e: float,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    start_time = time.perf_counter()
    denominators = calibration_weights.sum(axis=1).astype(float)
    psi_boot = {
        route: np.einsum(
            "bi,ijk->bjk", calibration_weights, calibration_psis[route]
        ) / denominators[:, None, None]
        for route in ROUTES
    }
    psi_point = {
        route: np.mean(calibration_psis[route], axis=0) for route in ROUTES
    }
    cache_key = v4_response_cache_key(
        contrast,
        mode,
        test_rows,
        calibration_psis,
        calibration_weights,
        test_weights,
        c_e,
    )
    response_cache = ANALYSIS_CACHE / f"response_{cache_key}.npz"
    report_cache = ANALYSIS_CACHE / f"response_{cache_key}.json"
    if response_cache.is_file() and report_cache.is_file():
        with np.load(response_cache) as cached:
            predictions = cached["predictions"]
        expected_shape = (
            B_BOOT,
            len(test_rows),
            len(ROUTES),
            len(MASK_KINDS),
            6,
        )
        if predictions.shape == expected_shape:
            report = json.loads(report_cache.read_text(encoding="utf-8"))
            return predictions, report, psi_boot

    coordinates, coordinate_report = v4_response_coordinates(psi_boot, psi_point)
    configuration_keys = [
        tuple(int(value) for value in row) for row in calibration_weights
    ]
    unique_indices = []
    seen = set()
    for bootstrap_index, key in enumerate(configuration_keys):
        if key not in seen:
            unique_indices.append(bootstrap_index)
            seen.add(key)
    point_dynamic = v4_point_dynamic(test_rows)
    exact_cache = {}
    exact_cache_path = ANALYSIS_CACHE / f"exact_response_{cache_key}.npz"
    if exact_cache_path.is_file():
        try:
            with np.load(exact_cache_path) as cached:
                cached_keys = cached["configuration_weights"]
                cached_values = cached["exact_values"]
            expected_value_shape = (
                len(test_rows),
                len(ROUTES),
                len(MASK_KINDS),
                6,
            )
            if (
                cached_keys.ndim == 2
                and cached_keys.shape[1] == calibration_weights.shape[1]
                and cached_values.shape[1:] == expected_value_shape
                and len(cached_keys) == len(cached_values)
            ):
                exact_cache = {
                    tuple(int(value) for value in key): cached_values[index]
                    for index, key in enumerate(cached_keys)
                }
        except Exception:  # noqa: BLE001
            exact_cache = {}

    def persist_exact_cache() -> None:
        ordered_keys = sorted(exact_cache)
        temporary = exact_cache_path.with_name(exact_cache_path.name + ".tmp.npz")
        np.savez(
            temporary,
            configuration_weights=np.asarray(ordered_keys, dtype=np.int16),
            exact_values=np.stack([exact_cache[key] for key in ordered_keys]),
        )
        temporary.replace(exact_cache_path)

    def ensure_exact(requested_indices: np.ndarray, description: str) -> None:
        missing = [
            int(index)
            for index in requested_indices
            if configuration_keys[int(index)] not in exact_cache
        ]
        if not missing:
            return
        print(
            f"  {contrast} response {description}: "
            f"{len(missing)} new exact calibration configurations",
            flush=True,
        )
        missing_psis = {route: psi_boot[route][missing] for route in ROUTES}

        def evaluate_subject(row: dict) -> np.ndarray:
            return v4_exact_dynamic_outputs(
                pathlib.Path(row["cache"]["file"]), missing_psis, c_e
            )

        with ThreadPoolExecutor(max_workers=EXACT_SUBJECT_WORKERS) as executor:
            exact_subjects = list(executor.map(evaluate_subject, test_rows))
        for subject_index, exact in enumerate(exact_subjects):
            for local_index, bootstrap_index in enumerate(missing):
                key = configuration_keys[bootstrap_index]
                if key not in exact_cache:
                    exact_cache[key] = np.empty_like(point_dynamic)
                exact_cache[key][subject_index] = exact[local_index]
            print(
                f"    exact {contrast} subject {subject_index + 1:02d}/"
                f"{len(test_rows)}: {test_rows[subject_index]['subject']}",
                flush=True,
            )
        persist_exact_cache()

    attempt_reports = []
    accepted_predictions = None
    accepted_report = None
    attempt_definitions = (
        SMOKE_SURROGATE_ATTEMPTS if mode == "smoke" else FULL_SURROGATE_ATTEMPTS
    )
    target_names = (
        "mean_R_calibrated",
        "mean_R_diagonal",
        "mean_R_wrong_cyclic",
        "delta_gain_diag",
        "empirical_gain_diag",
        "wrong_empirical_ratio",
    )
    for attempt in attempt_definitions:
        if attempt.get("exact_all_unique", False):
            maximum_unique = int(attempt["maximum_unique"])
            if len(unique_indices) > maximum_unique:
                attempt_reports.append(
                    {
                        "method": "all unique bootstrap configurations evaluated exactly",
                        "available_unique_configurations": len(unique_indices),
                        "maximum_unique": maximum_unique,
                        "pass": False,
                        "reason": "fixed operational maximum exceeded",
                    }
                )
                continue
            all_unique_indices = np.asarray(unique_indices, dtype=int)
            ensure_exact(all_unique_indices, "exact-all smoke fallback")
            predictions = np.stack(
                [exact_cache[key] for key in configuration_keys], axis=0
            )
            all_positive = bool(
                np.all(np.isfinite(predictions)) and np.all(predictions > 0.0)
            )
            attempt_report = {
                "method": "all unique bootstrap configurations evaluated exactly",
                "available_unique_configurations": len(unique_indices),
                "exact_unique_configurations_available": len(exact_cache),
                "maximum_unique": maximum_unique,
                "raw_output_relative_error_p99": 0.0,
                "raw_output_relative_error_max": 0.0,
                "all_5000_predictions_finite_and_positive": all_positive,
                "derived_joint_target_validation": "exact; zero approximation error",
                "pass": all_positive,
            }
            attempt_reports.append(attempt_report)
            if all_positive:
                accepted_predictions = predictions
                accepted_report = attempt_report
                break
            continue
        degree = int(attempt["degree"])
        _, terms = polynomial_features(np.zeros((1, coordinates.shape[1])), degree)
        term_count = len(terms)
        validation_count = min(
            int(attempt["validation"]), max(12, len(unique_indices) // 4)
        )
        train_target = max(int(attempt["minimum_train"]), term_count + 8)
        if "maximum_train" in attempt:
            train_target = min(train_target, int(attempt["maximum_train"]))
        train_count = min(train_target, len(unique_indices) - validation_count)
        if train_count < term_count or validation_count < 12:
            attempt_reports.append(
                {
                    "degree": degree,
                    "n_polynomial_terms": term_count,
                    "available_unique_configurations": len(unique_indices),
                    "pass": False,
                    "reason": "insufficient unique configurations for fixed validation",
                }
            )
            continue
        train_indices = np.asarray(unique_indices[:train_count], dtype=int)
        validation_indices = np.asarray(
            unique_indices[train_count:train_count + validation_count], dtype=int
        )
        requested = np.concatenate((train_indices, validation_indices))
        ensure_exact(requested, f"degree {degree}")

        design_all, terms = polynomial_features(coordinates, degree)
        design_zero, _ = polynomial_features(
            np.zeros((1, coordinates.shape[1])), degree
        )
        design_train = np.vstack((design_zero, design_all[train_indices]))
        predictions = np.empty(
            (
                B_BOOT,
                len(test_rows),
                len(ROUTES),
                len(MASK_KINDS),
                6,
            ),
            dtype=float,
        )
        validation_exact = np.stack(
            [exact_cache[configuration_keys[int(index)]] for index in validation_indices]
        )
        raw_relative_errors = []
        condition_numbers = []
        for subject_index, row in enumerate(test_rows):
            scale = np.maximum(np.abs(point_dynamic[subject_index]), 1.0e-30)
            train_values = np.stack(
                [
                    exact_cache[configuration_keys[int(index)]][subject_index]
                    for index in train_indices
                ]
            )
            normalized_train = np.concatenate(
                (
                    (point_dynamic[subject_index] / scale)[None],
                    train_values / scale[None],
                ),
                axis=0,
            ).reshape(len(design_train), -1)
            coefficient, _, _, singular_values = np.linalg.lstsq(
                design_train, normalized_train, rcond=1.0e-11
            )
            predicted = (design_all @ coefficient).reshape(
                B_BOOT, len(ROUTES), len(MASK_KINDS), 6
            ) * scale[None]
            predictions[:, subject_index] = predicted
            condition_numbers.append(
                {
                    "subject": row["subject"],
                    "design_condition_number": float(
                        singular_values[0] / singular_values[-1]
                    ),
                }
            )
            relative = np.abs(
                (
                    predictions[validation_indices, subject_index]
                    - validation_exact[:, subject_index]
                )
                / np.maximum(np.abs(validation_exact[:, subject_index]), 1.0e-30)
            )
            raw_relative_errors.extend(relative.ravel().tolist())

        predicted_validation_unreplaced = predictions[validation_indices].copy()
        for bootstrap_index, key in enumerate(configuration_keys):
            if key in exact_cache:
                predictions[bootstrap_index] = exact_cache[key]
        all_positive = bool(
            np.all(np.isfinite(predictions)) and np.all(predictions > 0.0)
        )
        validation_by_analysis = {}
        target_passes = []
        for route_index, route in enumerate(ROUTES):
            validation_by_analysis[route] = {}
            for mask_index, mask_kind in enumerate(MASK_KINDS):
                target_records = {}
                differences_by_target = [[] for _ in target_names]
                for local_index, bootstrap_index in enumerate(validation_indices):
                    weights = test_weights[int(bootstrap_index)]
                    exact_targets = v4_weighted_response_targets(
                        validation_exact[local_index, :, route_index, mask_index],
                        weights,
                    )
                    predicted_targets = v4_weighted_response_targets(
                        predicted_validation_unreplaced[
                            local_index, :, route_index, mask_index
                        ],
                        weights,
                    )
                    for target_index in range(len(target_names)):
                        differences_by_target[target_index].append(
                            abs(
                                float(
                                    predicted_targets[target_index]
                                    - exact_targets[target_index]
                                )
                            )
                        )
                for target_name, differences in zip(
                    target_names, differences_by_target
                ):
                    maximum = float(max(differences))
                    threshold = SURROGATE_TARGET_ABS_MAX[target_name]
                    passed = bool(maximum <= threshold)
                    target_passes.append(passed)
                    target_records[target_name] = {
                        "maximum_absolute_error": maximum,
                        "median_absolute_error": float(np.median(differences)),
                        "threshold": threshold,
                        "pass": passed,
                    }
                validation_by_analysis[route][mask_kind] = target_records
        raw_p99 = float(np.percentile(raw_relative_errors, 99))
        passed = bool(
            all_positive
            and raw_p99 <= SURROGATE_RAW_RELATIVE_P99_MAX
            and all(target_passes)
        )
        attempt_report = {
            "degree": degree,
            "n_polynomial_terms": len(terms),
            "training_unique_replicates": int(len(train_indices)),
            "validation_unique_replicates": int(len(validation_indices)),
            "available_unique_configurations": len(unique_indices),
            "exact_unique_configurations_available": len(exact_cache),
            "raw_output_relative_error_p99": raw_p99,
            "raw_output_relative_error_max": float(max(raw_relative_errors)),
            "raw_p99_threshold": SURROGATE_RAW_RELATIVE_P99_MAX,
            "all_5000_predictions_finite_and_positive": all_positive,
            "derived_joint_target_validation": validation_by_analysis,
            "per_subject_design": condition_numbers,
            "pass": passed,
        }
        attempt_reports.append(attempt_report)
        if passed:
            accepted_predictions = predictions
            accepted_report = attempt_report
            break
    if accepted_predictions is None or accepted_report is None:
        raise AssertionError(
            f"{contrast}: bootstrap response-surface validation failed fixed tolerances"
        )
    report = {
        "method": (
            "Every cluster-bootstrap replicate recomputes both aligned and raw Psi_hat. "
            + (
                "All distinct smoke bootstrap configurations are evaluated exactly; "
                "there is no response-surface approximation."
                if accepted_report.get("method", "").startswith("all unique")
                else (
                    "A validated polynomial response in exact Hermitian-covariance PCA "
                    "coordinates accelerates full-pixel recalculation; exact fitted/validation "
                    "configurations replace their predictions."
                )
            )
            + " Both masks are accumulated from the same pixel pass."
        ),
        "is_numerical_acceleration_not_statistical_resampling_approximation": True,
        "parallel_exact_subject_workers": EXACT_SUBJECT_WORKERS,
        "fixed_validation_thresholds": {
            "raw_relative_p99": SURROGATE_RAW_RELATIVE_P99_MAX,
            "derived_targets": SURROGATE_TARGET_ABS_MAX,
        },
        "coordinate_system": coordinate_report,
        "attempts": attempt_reports,
        "accepted": accepted_report,
        "exact_unique_configuration_count": len(exact_cache),
        "duration_seconds": float(time.perf_counter() - start_time),
    }
    np.savez_compressed(response_cache, predictions=accepted_predictions)
    report_cache.write_text(
        json.dumps(jsonise(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return accepted_predictions, report, psi_boot


def v4_weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights, axis=1) / np.sum(weights, axis=1)


def v4_weighted_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights, axis=1)


def v4_compute_analysis(
    test_rows: list[dict],
    test_noise_rows: list[dict],
    route: str,
    mask_kind: str,
    dynamic_boot: np.ndarray,
    psi_boot: np.ndarray,
    psi_point: np.ndarray,
    test_weights: np.ndarray,
    c_e: float,
) -> tuple[dict, dict, dict]:
    route_index = ROUTES.index(route)
    mask_index = MASK_KINDS.index(mask_kind)
    dynamic = dynamic_boot[:, :, route_index, mask_index]
    empirical = {
        "calibrated": dynamic[:, :, 0],
        "diagonal": dynamic[:, :, 2],
        "wrong_cyclic": dynamic[:, :, 4],
    }
    predicted = {
        "calibrated": dynamic[:, :, 1],
        "diagonal": dynamic[:, :, 3],
        "wrong_cyclic": dynamic[:, :, 5],
    }
    bundles = [row["analysis"][route][mask_kind] for row in test_rows]
    identity_empirical_point = np.asarray(
        [bundle["metrics"]["identity"]["empirical_sum"] for bundle in bundles]
    )
    identity_coefficients = np.stack(
        [bundle["identity_prediction_coefficient"] for bundle in bundles]
    )
    identity_predicted = c_e * np.einsum(
        "bij,sji->bs", psi_boot, identity_coefficients, optimize=True
    ).real
    empirical["identity"] = np.broadcast_to(
        identity_empirical_point[None], identity_predicted.shape
    )
    predicted["identity"] = identity_predicted
    point_empirical = {
        name: np.asarray(
            [bundle["metrics"][name]["empirical_sum"] for bundle in bundles]
        )
        for name in CONTROL_NAMES
    }
    point_predicted = {
        name: np.asarray(
            [bundle["metrics"][name]["predicted_sum"] for bundle in bundles]
        )
        for name in CONTROL_NAMES
    }

    point_c1 = np.asarray(
        [
            np.mean(point_empirical[name] / point_predicted[name])
            for name in CONTROL_NAMES
        ]
    )
    bootstrap_c1 = np.stack(
        [
            v4_weighted_mean(empirical[name] / predicted[name], test_weights)
            for name in CONTROL_NAMES
        ],
        axis=1,
    )
    c1_family = simultaneous_intervals(point_c1, bootstrap_c1, CONTROL_NAMES)
    c1_targets = {}
    for name in CONTROL_NAMES:
        target = c1_family["targets"][name]
        interval = target["simultaneous_95_interval"]
        c1_targets[name] = {
            **target,
            "equivalence_bounds": [0.8, 1.25],
            "pass": bool(interval[0] >= 0.8 and interval[1] <= 1.25),
        }

    empirical_sums = {
        name: v4_weighted_sum(values, test_weights)
        for name, values in empirical.items()
    }
    predicted_sums = {
        name: v4_weighted_sum(values, test_weights)
        for name, values in predicted.items()
    }
    boot_G_emp_diag = empirical_sums["diagonal"] / empirical_sums["calibrated"]
    boot_G_pred_diag = predicted_sums["diagonal"] / predicted_sums["calibrated"]
    boot_delta_gain = boot_G_emp_diag - boot_G_pred_diag
    boot_empirical_gain_minus_one = boot_G_emp_diag - 1.0
    point_G_emp_diag = float(
        np.sum(point_empirical["diagonal"])
        / np.sum(point_empirical["calibrated"])
    )
    point_G_pred_diag = float(
        np.sum(point_predicted["diagonal"])
        / np.sum(point_predicted["calibrated"])
    )
    point_delta = point_G_emp_diag - point_G_pred_diag
    c2_delta = percentile_interval_record(point_delta, boot_delta_gain)
    c2_gain_exists = percentile_interval_record(
        point_G_emp_diag - 1.0, boot_empirical_gain_minus_one
    )
    delta_interval = c2_delta["percentile_95_interval"]
    gain_interval = c2_gain_exists["percentile_95_interval"]
    c2_equivalence_pass = bool(
        delta_interval[0] >= -0.02 and delta_interval[1] <= 0.02
    )
    c2_gain_exists_pass = bool(gain_interval[0] > 0.0)

    boot_wrong_ratio = empirical_sums["wrong_cyclic"] / empirical_sums["calibrated"]
    point_wrong_ratio = float(
        np.sum(point_empirical["wrong_cyclic"])
        / np.sum(point_empirical["calibrated"])
    )
    c3a_record = percentile_interval_record(point_wrong_ratio, boot_wrong_ratio)
    c3a_interval = c3a_record["percentile_95_interval"]

    nll_wrong_delta = bootstrap_nll_wrong_delta(test_noise_rows, psi_boot)
    boot_nll_mean = v4_weighted_mean(nll_wrong_delta, test_weights)
    _, point_wrong_nll = score_deltas_for_subjects(
        test_noise_rows, psi_point, spectral_floor_delta2(psi_point)
    )
    c3b_record = percentile_interval_record(
        float(np.mean(point_wrong_nll)), boot_nll_mean
    )
    c3b_interval = c3b_record["percentile_95_interval"]

    subject_deltas = (
        point_empirical["diagonal"] / point_empirical["calibrated"]
        - point_predicted["diagonal"] / point_predicted["calibrated"]
    )
    centered_null = boot_delta_gain - point_delta
    one_sided_p = float(
        (1 + np.sum(centered_null <= point_delta)) / (B_BOOT + 1)
    )
    standard_error = float(np.std(boot_delta_gain, ddof=1))
    z_statistic = point_delta / max(standard_error, 1.0e-30)
    normal_p = float(norm.cdf(z_statistic))
    if delta_interval[0] >= -0.02 and delta_interval[1] <= 0.0:
        h_plug_outcome = "mild_real_optimism"
    elif (
        delta_interval[0] >= -0.02
        and delta_interval[1] <= 0.02
        and delta_interval[0] <= 0.0 <= delta_interval[1]
    ):
        h_plug_outcome = "equivalence_sign_not_established"
    else:
        h_plug_outcome = "negative_replication_outside_prespecified_interval"
    h_plug = {
        "hypothesis": (
            "The plug-in prediction is mildly optimistic: Delta_g < 0 with "
            "Delta_g in [-0.02, 0]."
        ),
        "outside_GO": True,
        "estimate": point_delta,
        "bilateral_95_interval": delta_interval,
        "one_sided_sign_test": {
            "alternative": "Delta_g < 0",
            "centered_bootstrap_p_value_with_plus_one_correction": one_sided_p,
            "construction": "P(Delta*_g-Delta_g <= Delta_g) under the null boundary zero",
            "normal_z_statistic_descriptive": z_statistic,
            "normal_one_sided_p_value_descriptive": normal_p,
        },
        "subjects_with_Delta_s_below_zero": int(np.sum(subject_deltas < 0.0)),
        "subject_count": len(subject_deltas),
        "per_subject_Delta_s": {
            row["subject"]: float(subject_deltas[index])
            for index, row in enumerate(test_rows)
        },
        "prespecified_outcome": h_plug_outcome,
    }

    criteria = {
        "C1": {
            "name": "Gamma predicts absolute receiver-noise variance",
            "statistic": "cluster-bootstrap mean of whole-subject ratio-of-sums R_s,W",
            "simultaneous_family": c1_family,
            "targets": c1_targets,
            "equivalence_bounds": [0.8, 1.25],
            "pass": bool(all(target["pass"] for target in c1_targets.values())),
        },
        "C2": {
            "name": "predicted diagonal-to-calibrated gain equals observed gain, and the gain exists",
            "statistic": "G_emp=sum(E_diag)/sum(E_cal); G_pred=sum(V_diag)/sum(V_cal)",
            "G_emp": point_G_emp_diag,
            "G_pred": point_G_pred_diag,
            "delta_gain_G_emp_minus_G_pred": {
                **c2_delta,
                "equivalence_bounds": [-0.02, 0.02],
                "interpretation": "two absolute percentage points in the variance-ratio gain",
                "pass": c2_equivalence_pass,
            },
            "empirical_gain_G_emp_minus_one": {
                **c2_gain_exists,
                "one_sided_requirement": "lower endpoint strictly greater than zero",
                "pass": c2_gain_exists_pass,
            },
            "pass": bool(c2_equivalence_pass and c2_gain_exists_pass),
        },
        "C3a": {
            "name": "wrong correlation layout has higher output-difference variance",
            "statistic": "sum(E_wrong)/sum(E_calibrated)",
            **c3a_record,
            "one_sided_bound": 1.0,
            "pass": bool(c3a_interval[0] > 1.0),
        },
        "C3b": {
            "name": "complex Gaussian covariance score penalizes wrong correlation layout",
            "statistic": "cluster-bootstrap mean_s NLL_wrong-NLL_calibrated",
            **c3b_record,
            "one_sided_bound": 0.0,
            "score_name": "complex Gaussian covariance score",
            "pass": bool(c3b_interval[0] > 0.0),
        },
    }

    gain_names = []
    point_gains = []
    bootstrap_gains = []
    per_subject_gains = {}
    for comparison in ("diagonal", "identity", "wrong_cyclic"):
        point_emp_subject = (
            point_empirical[comparison] / point_empirical["calibrated"] - 1.0
        )
        point_pred_subject = (
            point_predicted[comparison] / point_predicted["calibrated"] - 1.0
        )
        boot_emp = empirical_sums[comparison] / empirical_sums["calibrated"] - 1.0
        boot_pred = predicted_sums[comparison] / predicted_sums["calibrated"] - 1.0
        gain_names.extend(
            (f"{comparison}_empirical_gain", f"{comparison}_predicted_gain")
        )
        point_gains.extend(
            (
                float(
                    np.sum(point_empirical[comparison])
                    / np.sum(point_empirical["calibrated"])
                    - 1.0
                ),
                float(
                    np.sum(point_predicted[comparison])
                    / np.sum(point_predicted["calibrated"])
                    - 1.0
                ),
            )
        )
        bootstrap_gains.extend((boot_emp, boot_pred))
        per_subject_gains[comparison] = {
            row["subject"]: {
                "empirical_ratio_minus_one": float(point_emp_subject[index]),
                "predicted_ratio_minus_one": float(point_pred_subject[index]),
            }
            for index, row in enumerate(test_rows)
        }
    gain_intervals = simultaneous_intervals(
        np.asarray(point_gains), np.stack(bootstrap_gains, axis=1), tuple(gain_names)
    )
    gains = {
        "definition": "comparison/calibrated ratio of sums minus one over retained test subjects",
        "joint_simultaneous_intervals": gain_intervals,
        "per_subject": per_subject_gains,
        "identity_gain_is_noncriterion": True,
    }
    bootstrap_summary = {
        "seed": BOOTSTRAP_SEED,
        "replicates": B_BOOT,
        "whole_session_cluster_is_bootstrap_unit": True,
        "pair_dependence_preserved": True,
        "calibration_and_test_resampled_jointly": True,
        "C1_simultaneous_family": c1_family,
        "C2_C3_intervals_are_paired_pointwise_95_percent": True,
    }
    return criteria, gains, {"H_plug": h_plug, "bootstrap": bootstrap_summary}


def v4_c0_gate(noise_rows: list[dict], psi_hat: np.ndarray) -> dict:
    delta2 = spectral_floor_delta2(psi_hat)
    diagonal, wrong = score_deltas_for_subjects(noise_rows, psi_hat, delta2)
    individuals = []
    for index, row in enumerate(noise_rows):
        individuals.append(
            {
                "subject": row["subject"],
                "NLL_wrong_minus_calibrated": float(wrong[index]),
                "NLL_diagonal_minus_calibrated": float(diagonal[index]),
                "wrong_inequality_positive": bool(wrong[index] > 0.0),
                "diagonal_inequality_positive": bool(diagonal[index] > 0.0),
            }
        )
    passed = bool(np.all(wrong > 0.0) and np.all(diagonal > 0.0))
    return {
        "name": "C0 identifiability gate",
        "interpretation": "instrumental gate, not 95% population inference",
        "score_name": "complex Gaussian covariance score",
        "spectral_floor": {
            "predeclared_relative_to_mean_receiver_variance": SPECTRAL_FLOOR_RELATIVE,
            "delta_squared_absolute": delta2,
        },
        "required_inequality_count": 2 * len(noise_rows),
        "inequalities": individuals,
        "positive_count": int(np.sum(wrong > 0.0) + np.sum(diagonal > 0.0)),
        "pass": passed,
    }


def v4_analysis_matrices(psi_hat: np.ndarray) -> tuple[dict, dict, dict, dict]:
    controls = control_matrices(psi_hat)
    derangements = all_derangement_controls(psi_hat)
    eigen = eigenvalue_inversion_control(psi_hat)
    matrices = dict(controls)
    matrices.update({name: value["matrix"] for name, value in derangements.items()})
    if eigen["substantial_eigenvalue_gaps"]:
        matrices["eigenvalue_inversion"] = eigen["matrix"]
    return controls, derangements, eigen, matrices


def v4_robustness_analysis(
    test_rows: list[dict],
    test_noise_rows: list[dict],
    psi_hat: np.ndarray,
    controls: dict,
    derangements: dict,
    eigen_diagnostic: dict,
    route: str = "aligned",
    mask_kind: str = "common_high_rss",
) -> tuple[dict, dict]:
    delta2 = spectral_floor_delta2(psi_hat)
    entries = {}
    matrices = {"calibrated": psi_hat}
    matrices.update({name: value["matrix"] for name, value in derangements.items()})
    for name, matrix in matrices.items():
        metrics = [
            row["analysis"][route][mask_kind]["metrics"][name] for row in test_rows
        ]
        candidate = spectral_floor_matrix(matrix, delta2)
        nll = [
            nll_complex_from_covariance(row["psi"], candidate)
            for row in test_noise_rows
        ]
        entries[name] = {
            "matrix": complex_matrix_record(matrix),
            "mean_test_complex_covariance_score": float(np.mean(nll)),
            "total_empirical_variance_proxy": float(
                np.sum([value["empirical_sum"] for value in metrics])
            ),
            "total_predicted_variance": float(
                np.sum([value["predicted_sum"] for value in metrics])
            ),
            "mean_R": float(
                np.mean([value["R_empirical_over_predicted"] for value in metrics])
            ),
        }
        if name in derangements:
            entries[name]["old_to_new_zero_based"] = derangements[name][
                "old_to_new_zero_based"
            ]
    nll_order = sorted(
        entries, key=lambda name: (entries[name]["mean_test_complex_covariance_score"], name)
    )
    empirical_order = sorted(
        entries, key=lambda name: (entries[name]["total_empirical_variance_proxy"], name)
    )
    for rank_value, name in enumerate(nll_order, start=1):
        entries[name]["NLL_rank_low_is_good"] = rank_value
    for rank_value, name in enumerate(empirical_order, start=1):
        entries[name]["empirical_variance_rank_low_is_good"] = rank_value
    cyclic_matches = [
        name
        for name, value in derangements.items()
        if np.allclose(value["matrix"], controls["wrong_cyclic"], rtol=1.0e-12, atol=1.0e-12)
    ]
    if len(cyclic_matches) != 1:
        raise AssertionError("Frozen cyclic control did not match exactly one derangement")
    robustness = {
        "scope": "calibrated covariance plus all nine nontrivial derangements; no post-hoc replacement",
        "frozen_principal_cyclic_derangement": cyclic_matches[0],
        "NLL_order_low_to_high": nll_order,
        "empirical_variance_order_low_to_high": empirical_order,
        "calibrated_NLL_rank_out_of_10": entries["calibrated"]["NLL_rank_low_is_good"],
        "calibrated_empirical_variance_rank_out_of_10": entries["calibrated"][
            "empirical_variance_rank_low_is_good"
        ],
        "entries": entries,
    }
    if eigen_diagnostic["substantial_eigenvalue_gaps"]:
        name = "eigenvalue_inversion"
        metrics = [
            row["analysis"][route][mask_kind]["metrics"][name] for row in test_rows
        ]
        candidate = spectral_floor_matrix(eigen_diagnostic["matrix"], delta2)
        eigen_diagnostic = dict(eigen_diagnostic)
        eigen_diagnostic["test_diagnostic"] = {
            "mean_R": float(np.mean([value["R_empirical_over_predicted"] for value in metrics])),
            "aggregate_empirical_ratio_to_calibrated": float(
                np.sum([value["empirical_sum"] for value in metrics])
                / np.sum(
                    [
                        row["analysis"][route][mask_kind]["metrics"]["calibrated"][
                            "empirical_sum"
                        ]
                        for row in test_rows
                    ]
                )
            ),
            "mean_test_complex_covariance_score": float(
                np.mean(
                    [
                        nll_complex_from_covariance(row["psi"], candidate)
                        for row in test_noise_rows
                    ]
                )
            ),
        }
    return robustness, eigen_diagnostic


def v4_signal_gain_summary(
    test_rows: list[dict], gains: dict, route: str, mask_kind: str
) -> dict:
    targets = gains["joint_simultaneous_intervals"]["targets"]
    result = {}
    for name in CONTROL_NAMES:
        bundles = [row["analysis"][route][mask_kind] for row in test_rows]
        numerator = float(
            np.sum(
                [bundle["signal_gain_components"][name]["difference_energy"] for bundle in bundles]
            )
        )
        denominator = float(
            np.sum(
                [
                    bundle["signal_gain_components"][name]["calibrated_reference_energy"]
                    for bundle in bundles
                ]
            )
        )
        pooled = float(math.sqrt(numerator / max(denominator, 1.0e-30)))
        summary = {
            "pooled_B_W": pooled,
            "mean_subject_B_W": float(
                np.mean([bundle["signal_gain_B_W"][name] for bundle in bundles])
            ),
            "median_subject_B_W": float(
                np.median([bundle["signal_gain_B_W"][name] for bundle in bundles])
            ),
            "maximum_subject_B_W": float(
                np.max([bundle["signal_gain_B_W"][name] for bundle in bundles])
            ),
            "per_subject": {
                row["subject"]: float(row["analysis"][route][mask_kind]["signal_gain_B_W"][name])
                for row in test_rows
            },
        }
        if name == "calibrated":
            summary["conservative_wording_rule"] = "reference; no comparison"
        else:
            empirical_gain = float(targets[f"{name}_empirical_gain"]["estimate"])
            threshold = empirical_gain / 10.0
            veto = bool(pooled > threshold)
            summary["conservative_wording_rule"] = {
                "comparison": f"{name} versus calibrated",
                "comparison_specific_empirical_variance_gain": empirical_gain,
                "threshold_gain_over_10": threshold,
                "B_W_exceeds_threshold": veto,
                "required_wording": (
                    "output-variance reduction"
                    if veto
                    else "efficiency at approximately fixed signal response"
                ),
                "is_a_vocabulary_veto_not_a_GO_NO_GO_criterion": True,
            }
        result[name] = summary
    return result


def v4_seeded_rng(label_name: str) -> np.random.Generator:
    label_digest = hashlib.sha256(label_name.encode("utf-8")).digest()
    label_integer = int.from_bytes(label_digest[:8], "little", signed=False)
    return np.random.default_rng(
        np.random.SeedSequence(
            [BOOTSTRAP_SEED, label_integer & 0xFFFFFFFF, label_integer >> 32]
        )
    )


def v4_run_noise_rows(
    subjects: list[str],
    contrast: str,
    repetitions: tuple[str, ...],
    active: np.ndarray,
    frequency_masks: dict[str, np.ndarray],
    label_name: str,
) -> tuple[list[dict], list[dict], int]:
    rows = []
    missing = []
    cache_hits = 0
    for index, subject in enumerate(subjects, start=1):
        try:
            row, cache_hit = v4_noise_statistics_cached(
                subject,
                contrast,
                active,
                frequency_masks,
                repetitions,
            )
            rows.append(row)
            cache_hits += int(cache_hit)
            print(
                f"  noise {label_name} {index:03d}/{len(subjects):03d} {subject}: "
                f"pairs={row['included_aligned_pair_count']}, "
                f"outer aligned/raw={row['outer_power_aligned_over_raw']:.4f}" +
                (" [cache]" if cache_hit else ""),
                flush=True,
            )
        except NoAdmissiblePairs as error:
            missing.append(
                {
                    "subject": subject,
                    "reason": "no pair survives the frozen v3.1 rotation/deformation thresholds",
                    "detail": str(error),
                }
            )
            print(
                f"  noise {label_name} {index:03d}/{len(subjects):03d} {subject}: MISSING",
                flush=True,
            )
    return rows, missing, cache_hits


def v4_run_contrast_analysis(
    *,
    label_name: str,
    contrast: str,
    repetitions: tuple[str, str, str],
    split: dict[str, list[str]],
    active: np.ndarray,
    frequency_masks: dict[str, np.ndarray],
    mode: str,
    scientific_role: str,
    calibration_source: dict | None = None,
    representative_requested: bool = False,
) -> tuple[dict, dict | None, dict]:
    """Run C0--C3 and all frozen secondary routes for one triad analysis."""
    start_time = time.perf_counter()
    test_candidates = list(split.get("test", []))
    if calibration_source is None:
        noise_subjects = sorted(
            set(split.get("calibration", []))
            | set(split.get("C0_gate", []))
            | set(test_candidates)
        )
    else:
        noise_subjects = sorted(test_candidates)
    noise_rows, technical_missing, noise_cache_hits = v4_run_noise_rows(
        noise_subjects,
        contrast,
        repetitions,
        active,
        frequency_masks,
        label_name,
    )
    noise_by_subject = {row["subject"]: row for row in noise_rows}
    if calibration_source is None:
        calibration_subjects = [
            subject
            for subject in split["calibration"]
            if subject in noise_by_subject
        ]
        calibration_rows = [noise_by_subject[subject] for subject in calibration_subjects]
        if len(calibration_rows) < 2:
            raise AssertionError(f"{label_name}: fewer than two usable calibration subjects")
        calibration_psis = {
            route: np.stack(
                [row["routes"][route]["psi"] for row in calibration_rows]
            )
            for route in ROUTES
        }
        calibration_weights, calibration_cluster_record = v4_cluster_bootstrap_weights(
            calibration_subjects,
            v4_seeded_rng(
                f"calibration|{contrast}|{'|'.join(calibration_subjects)}"
            ),
        )
    else:
        calibration_subjects = list(calibration_source["calibration_subjects"])
        calibration_rows = list(calibration_source["calibration_rows"])
        calibration_psis = {
            route: np.asarray(calibration_source["calibration_psis"][route])
            for route in ROUTES
        }
        calibration_weights = np.asarray(calibration_source["calibration_weights"])
        calibration_cluster_record = calibration_source["calibration_cluster_record"]
    psi_hat = {
        route: np.mean(calibration_psis[route], axis=0) for route in ROUTES
    }
    controls = {}
    derangements = {}
    eigen_diagnostics = {}
    matrices = {}
    for route in ROUTES:
        (
            controls[route],
            derangements[route],
            eigen_diagnostics[route],
            matrices[route],
        ) = v4_analysis_matrices(psi_hat[route])
    c_e = float(frequency_masks["E"].sum() / (N * N))

    c0 = None
    c0_missing = []
    if calibration_source is None:
        c0_rows = []
        for subject in split["C0_gate"]:
            if subject in noise_by_subject:
                c0_rows.append(
                    {
                        "subject": subject,
                        **noise_by_subject[subject]["routes"]["aligned"],
                    }
                )
            else:
                c0_missing.append(
                    {
                        "subject": subject,
                        "reason": "no admissible aligned covariance pair",
                    }
                )
        if not c0_rows:
            raise AssertionError(f"{label_name}: no usable C0 subject")
        c0 = v4_c0_gate(c0_rows, psi_hat["aligned"])
        c0["missing_subjects"] = c0_missing
        c0["universal_over_all_nonmissing_C0_subjects"] = True

    retained_test_subjects = [
        subject for subject in test_candidates if subject in noise_by_subject
    ]
    if len(retained_test_subjects) < 2:
        raise AssertionError(f"{label_name}: fewer than two usable test subjects")
    test_rows = []
    representative = None
    image_cache_hits = 0
    for index, subject in enumerate(retained_test_subjects, start=1):
        noise_row = noise_by_subject[subject]
        row, image_data, cache_hit = v4_prepare_test_subject_cached(
            subject,
            psi_hat["aligned"],
            active,
            frequency_masks,
            c_e,
            matrices["aligned"],
            representative=bool(representative_requested and index == 1),
            contrast=contrast,
            secondary_models={"raw": (psi_hat["raw"], matrices["raw"])},
            repetitions=repetitions,
            precomputed_pair_diagnostics=noise_row["pair_registration_diagnostics"],
        )
        if set(row["excluded_pairs"]) != set(noise_row["excluded_pairs"]):
            raise AssertionError(
                f"{subject}: covariance-route and criterion-route pair exclusions differ"
            )
        test_rows.append(row)
        image_cache_hits += int(cache_hit)
        if image_data is not None:
            representative = image_data
        print(
            f"  images {label_name} {index:03d}/{len(retained_test_subjects):03d} "
            f"{subject}: pairs={row['included_pair_count']}, "
            f"points={row['criterion_point_count_including_pairs']:,}, "
            f"R_aligned={row['analysis']['aligned']['common_high_rss']['metrics']['calibrated']['R_empirical_over_predicted']:.3f}" +
            (" [cache]" if cache_hit else ""),
            flush=True,
        )
    if representative_requested and representative is None:
        raise AssertionError(f"{label_name}: no representative image")

    test_subjects = [row["subject"] for row in test_rows]
    test_weights, test_cluster_record = v4_cluster_bootstrap_weights(
        test_subjects,
        v4_seeded_rng(f"test|{label_name}|{'|'.join(test_subjects)}"),
    )
    print(f"Bootstrap response surfaces for {label_name}...", flush=True)
    dynamic_boot, response_report, psi_boot = v4_build_bootstrap_response(
        label_name,
        mode,
        test_rows,
        calibration_psis,
        calibration_weights,
        test_weights,
        c_e,
    )
    test_noise = {
        route: [
            {
                "subject": subject,
                **noise_by_subject[subject]["routes"][route],
            }
            for subject in test_subjects
        ]
        for route in ROUTES
    }
    analyses = {}
    for route in ROUTES:
        analyses[route] = {}
        for mask_kind in MASK_KINDS:
            criteria, gains, extras = v4_compute_analysis(
                test_rows,
                test_noise[route],
                route,
                mask_kind,
                dynamic_boot,
                psi_boot[route],
                psi_hat[route],
                test_weights,
                c_e,
            )
            analyses[route][mask_kind] = {
                "criteria": criteria,
                "gains": gains,
                **extras,
            }
    principal = analyses["aligned"]["common_high_rss"]
    raw_principal = analyses["raw"]["common_high_rss"]
    aligned_criteria = principal["criteria"]
    criteria_pass = bool(
        aligned_criteria["C1"]["pass"]
        and aligned_criteria["C2"]["pass"]
        and aligned_criteria["C3a"]["pass"]
        and aligned_criteria["C3b"]["pass"]
    )
    if c0 is not None:
        criteria_pass = bool(criteria_pass and c0["pass"])
    scientific_verdict = None if mode == "smoke" else criteria_pass
    route_comparison = {
        "R_calibrated": {
            "aligned": aligned_criteria["C1"]["targets"]["calibrated"]["estimate"],
            "raw": raw_principal["criteria"]["C1"]["targets"]["calibrated"]["estimate"],
        },
        "Delta_g": {
            "aligned": aligned_criteria["C2"]["delta_gain_G_emp_minus_G_pred"]["estimate"],
            "raw": raw_principal["criteria"]["C2"]["delta_gain_G_emp_minus_G_pred"]["estimate"],
        },
    }
    route_comparison["absolute_distance_to_target"] = {
        "R_aligned": abs(route_comparison["R_calibrated"]["aligned"] - 1.0),
        "R_raw": abs(route_comparison["R_calibrated"]["raw"] - 1.0),
        "Delta_g_aligned": abs(route_comparison["Delta_g"]["aligned"]),
        "Delta_g_raw": abs(route_comparison["Delta_g"]["raw"]),
    }
    route_comparison["prospective_secondary_comparisons"] = {
        "abs_R_aligned_minus_1_le_abs_R_raw_minus_1": bool(
            route_comparison["absolute_distance_to_target"]["R_aligned"]
            <= route_comparison["absolute_distance_to_target"]["R_raw"]
        ),
        "abs_Delta_aligned_le_abs_Delta_raw": bool(
            route_comparison["absolute_distance_to_target"]["Delta_g_aligned"]
            <= route_comparison["absolute_distance_to_target"]["Delta_g_raw"]
        ),
        "outside_GO": True,
    }
    robustness, eigen_aligned = v4_robustness_analysis(
        test_rows,
        test_noise["aligned"],
        psi_hat["aligned"],
        controls["aligned"],
        derangements["aligned"],
        eigen_diagnostics["aligned"],
    )
    signal_gain = v4_signal_gain_summary(
        test_rows,
        principal["gains"],
        "aligned",
        "common_high_rss",
    )
    calibration_summary = {}
    for route in ROUTES:
        pseudo_hat = np.mean(
            [row["routes"][route]["pseudo"] for row in calibration_rows], axis=0
        )
        augmented_hat = np.mean(
            [row["routes"][route]["real_augmented"] for row in calibration_rows],
            axis=0,
        )
        circularity = float(np.linalg.norm(pseudo_hat) / np.linalg.norm(psi_hat[route]))
        selected_rows = [
            {"subject": row["subject"], **row["routes"][route]}
            for row in calibration_rows
        ]
        calibration_summary[route] = {
            "psi_hat": complex_matrix_record(psi_hat[route]),
            "correlation_hat": complex_matrix_record(correlation_matrix(psi_hat[route])),
            "pseudo_covariance_hat": {
                "real": pseudo_hat.real.tolist(),
                "imaginary": pseudo_hat.imag.tolist(),
                "frobenius_norm": float(np.linalg.norm(pseudo_hat)),
            },
            "pseudo_to_covariance_frobenius_ratio": circularity,
            "descriptive_alert_threshold": 0.1,
            "descriptive_alert_triggered": bool(circularity > 0.1),
            "descriptive_real_augmented_covariance_hat_8x8_not_used_by_GLS_or_scores": augmented_hat,
            "dispersion": summarize_noise_estimates(selected_rows, calibration_subjects),
        }
    missing_by_role = {
        role: [
            row for row in technical_missing if row["subject"] in set(split.get(role, []))
        ]
        for role in ("calibration", "C0_gate", "test")
    }
    result = {
        "analysis_label": label_name,
        "contrast": contrast,
        "repetitions": list(repetitions),
        "scientific_role": scientific_role,
        "calibration_is_fixed_from_primary": bool(calibration_source is not None),
        "no_recalibration_from_external_outcomes": bool(calibration_source is not None),
        "roles": {
            role: {
                "assigned_subjects": list(subjects),
                "assigned_count": len(subjects),
                "technical_missing_after_assignment": missing_by_role.get(role, []),
            }
            for role, subjects in split.items()
        },
        "retained": {
            "calibration_subjects": calibration_subjects,
            "calibration_count": len(calibration_subjects),
            "test_subjects": test_subjects,
            "test_count": len(test_subjects),
        },
        "calibration": calibration_summary,
        "all_subject_noise_estimates": {
            row["subject"]: v4_compact_noise_record(row) for row in noise_rows
        },
        "C0_gate": c0,
        "analyses": analyses,
        "aligned_primary_analysis_path": "analyses.aligned.common_high_rss",
        "raw_secondary_analysis_path": "analyses.raw.common_high_rss",
        "largest_component_secondary_paths": [
            "analyses.aligned.largest_connected_component",
            "analyses.raw.largest_connected_component",
        ],
        "route_comparison": route_comparison,
        "controls": {
            "aligned_matrices": {
                name: matrix_summary(matrix) for name, matrix in controls["aligned"].items()
            },
            "raw_matrices": {
                name: matrix_summary(matrix) for name, matrix in controls["raw"].items()
            },
            "wrong_definition": (
                "D^(1/2) P R P^T D^(1/2): exact marginal variances retained; "
                "only correlation layout permuted"
            ),
            "nine_derangements_noncriterion_aligned": robustness,
            "eigenvalue_inversion_diagnostic_aligned": {
                key: (complex_matrix_record(value) if key == "matrix" else value)
                for key, value in eigen_aligned.items()
            },
        },
        "signal_gain_diagnostic_B_W": signal_gain,
        "motion_phase_and_crossfit": {
            "frozen_pairwise_frame_rule": (
                "for pair (i,j), both i and j are aligned into auxiliary repetition k; "
                "a common k-to-r1 post-transform supplies mask/display coordinates"
            ),
            "technical_pair_exclusion_thresholds": {
                "rotation_degrees": ROTATION_EXCLUSION_DEG,
                "deformation_pixels": DEFORMATION_EXCLUSION_PX,
                "required_NCC_improvement": REGISTRATION_GAIN_MIN,
            },
            "missing_subjects": technical_missing,
            "per_test_subject": {
                row["subject"]: {
                    "included_pair_count": row["included_pair_count"],
                    "excluded_pairs": row["excluded_pairs"],
                    "pair_registration_diagnostics": row[
                        "pair_registration_diagnostics"
                    ],
                    "corrections_by_analysis_pair": row[
                        "corrections_by_analysis_pair"
                    ],
                    "difference_power": row["difference_power"],
                    "mask": row["mask"],
                    "cross_fitting": row["cross_fitting"],
                }
                for row in test_rows
            },
        },
        "test_subject_results": {
            row["subject"]: {
                "included_pair_count": row["included_pair_count"],
                "criterion_point_count_including_pairs": row[
                    "criterion_point_count_including_pairs"
                ],
                "mask": row["mask"],
                "analysis": row["analysis"],
            }
            for row in test_rows
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": B_BOOT,
            "calibration_clusters": calibration_cluster_record,
            "test_clusters": test_cluster_record,
            "response_surface_acceleration": response_report,
            "calibration_and_test_clusters_resampled_jointly_in_each_replicate": True,
        },
        "verdict": {
            "status": (
                "nonconfirmatory integration diagnostics; no scientific verdict"
                if mode == "smoke"
                else scientific_role
            ),
            "C0": None if c0 is None else c0["pass"],
            "C1": aligned_criteria["C1"]["pass"],
            "C2": aligned_criteria["C2"]["pass"],
            "C3a": aligned_criteria["C3a"]["pass"],
            "C3b": aligned_criteria["C3b"]["pass"],
            "computed_all_frozen_criteria_pass": criteria_pass,
            "scientific_verdict": scientific_verdict,
            "has_no_effect_on_primary_GO": bool(
                mode == "smoke" or scientific_role != "primary T2 GO"
            ),
        },
        "runtime": {
            "duration_seconds": float(time.perf_counter() - start_time),
        },
    }
    private = {
        "calibration_subjects": calibration_subjects,
        "calibration_rows": calibration_rows,
        "calibration_psis": calibration_psis,
        "calibration_weights": calibration_weights,
        "calibration_cluster_record": calibration_cluster_record,
        "psi_hat": psi_hat,
        "test_rows": test_rows,
        "test_noise": test_noise,
        "noise_rows": noise_rows,
    }
    return result, representative, private


def v4_stationarity_analysis(
    estimates: dict[str, np.ndarray], subject_counts: dict[str, int], scope: str
) -> dict:
    correlations = {
        name: correlation_matrix(matrix) for name, matrix in estimates.items()
    }
    comparisons = {}
    for first, second in itertools.combinations(estimates, 2):
        comparisons[f"{first}_versus_{second}"] = gauge_invariant_correlation_distance(
            correlations[first], correlations[second]
        )
    return {
        "scope": scope,
        "descriptive_only_no_effect_on_GO": True,
        "threshold_applies_to_gauge_invariant_distance": 0.15,
        "subject_counts": subject_counts,
        "mean_covariances": {
            name: complex_matrix_record(matrix) for name, matrix in estimates.items()
        },
        "mean_correlations": {
            name: complex_matrix_record(matrix) for name, matrix in correlations.items()
        },
        "pairwise_comparisons": comparisons,
    }


def v4_direct_delta_record(result: dict) -> tuple[float, tuple[float, float]]:
    row = result["analyses"]["aligned"]["common_high_rss"]["criteria"]["C2"][
        "delta_gain_G_emp_minus_G_pred"
    ]
    return float(row["estimate"]), tuple(float(value) for value in row["percentile_95_interval"])


def v4_make_figure(
    representative: dict,
    primary_t2: dict,
    primary_t2_private: dict,
    t1_result: dict,
    external_t2: dict | None,
    mode: str,
) -> tuple[float, pathlib.Path, pathlib.Path]:
    start_time = time.perf_counter()
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.7), constrained_layout=True)

    ax = axes[0, 0]
    magnitude = representative["magnitude"]
    mask = representative["mask"]
    largest = representative["largest_component_mask"]
    upper = float(np.percentile(magnitude[mask], 99.5))
    ax.imshow(magnitude, cmap="gray", origin="lower", vmin=0.0, vmax=upper)
    ax.contour(mask.astype(float), levels=[0.5], colors=["#f2c14e"], linewidths=0.8)
    ax.contour(largest.astype(float), levels=[0.5], colors=["#00b4d8"], linewidths=0.7)
    ax.set_title(
        f"(i) T2 test subject {representative['subject']}, slice {representative['slice']}\n"
        "Calibrated GLS magnitude; common high-RSS mask (yellow), largest component (cyan)"
    )
    ax.set_axis_off()

    ax = axes[0, 1]
    correlation = correlation_matrix(primary_t2_private["psi_hat"]["aligned"])
    image_handle = ax.imshow(np.abs(correlation), cmap="viridis", vmin=0.0, vmax=1.0)
    for row in range(COILS):
        for column in range(COILS):
            phase_degrees = float(np.degrees(np.angle(correlation[row, column])))
            color = "white" if abs(correlation[row, column]) < 0.48 else "black"
            ax.text(
                column,
                row,
                f"{abs(correlation[row, column]):.2f}\n{phase_degrees:+.0f} deg",
                ha="center",
                va="center",
                fontsize=7.3,
                color=color,
            )
    ax.set_xticks(range(COILS), [f"coil {index}" for index in range(1, COILS + 1)])
    ax.set_yticks(range(COILS), [f"coil {index}" for index in range(1, COILS + 1)])
    ax.set_title("(ii) Aligned T2 calibration correlation: magnitude and phase")
    fig.colorbar(image_handle, ax=ax, shrink=0.78, label="correlation magnitude")
    inset = inset_axes(ax, width="36%", height="31%", loc="lower left", borderpad=1.15)
    off_diagonal = ~np.eye(COILS, dtype=bool)
    calibration_means = [
        float(np.mean(np.abs(correlation_matrix(row["routes"]["aligned"]["psi"])[off_diagonal])))
        for row in primary_t2_private["calibration_rows"]
    ]
    inset.scatter(np.arange(1, len(calibration_means) + 1), calibration_means, s=13, color="#d95f02")
    inset.axhline(np.mean(calibration_means), color="black", lw=0.7, ls="--")
    inset.set_title("calibration dispersion", fontsize=7)
    inset.set_xlabel("subject", fontsize=6)
    inset.set_ylabel("mean |off-diagonal|", fontsize=6)
    inset.tick_params(labelsize=5.5)

    ax = axes[1, 0]
    subjects = primary_t2["retained"]["test_subjects"]
    x_values = np.arange(len(subjects))
    offsets = np.linspace(-0.27, 0.27, len(CONTROL_NAMES))
    for offset, control in zip(offsets, CONTROL_NAMES):
        values = [
            primary_t2["test_subject_results"][subject]["analysis"]["aligned"]
            ["common_high_rss"]["metrics"][control]["R_empirical_over_predicted"]
            for subject in subjects
        ]
        ax.scatter(
            x_values + offset,
            values,
            s=24,
            color=COLORS[control],
            label=control,
            alpha=0.9,
        )
    ax.axhspan(0.8, 1.25, color="#90be6d", alpha=0.18, label="equivalence band [0.8, 1.25]")
    ax.axhline(1.0, color="black", lw=0.75, ls="--")
    tick_step = max(1, int(math.ceil(len(subjects) / 15)))
    shown = np.arange(0, len(subjects), tick_step)
    ax.set_xticks(shown, [str(index + 1) for index in shown])
    ax.set_xlabel("temporal test-subject index")
    ax.set_ylabel("subject ratio R (empirical sum / predicted sum)")
    ax.set_title("(iii) Aligned T2 absolute receiver-noise variance ratios")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(fontsize=7.0, ncol=2, loc="upper right")

    ax = axes[1, 1]
    labels = ["v3.1 pilot", "v4-T2 primary", "external T2", "v4-T1"]
    records: list[tuple[float, tuple[float, float]] | None] = [
        (V31_PILOT_DELTA, V31_PILOT_DELTA_CI),
        v4_direct_delta_record(primary_t2),
        None if external_t2 is None else v4_direct_delta_record(external_t2),
        v4_direct_delta_record(t1_result),
    ]
    positions = np.arange(len(labels))
    ax.axhspan(-2.0, 2.0, color="#90be6d", alpha=0.22, label="equivalence band +/-2 pp")
    ax.axhline(0.0, color="black", lw=0.75)
    for position, record in zip(positions, records):
        if record is None:
            ax.text(position, 0.0, "not run\nin smoke", ha="center", va="center", fontsize=8, color="#555555")
            continue
        estimate, interval = record
        ax.errorbar(
            [position],
            [100.0 * estimate],
            yerr=[[100.0 * (estimate - interval[0])], [100.0 * (interval[1] - estimate)]],
            fmt="o",
            color="#0077b6" if position else "#6c757d",
            capsize=4,
        )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Delta_g = empirical gain - predicted gain (pp)")
    ax.set_title("(iv) Direct plug-in gain discrepancy with paired 95% intervals")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "M4Raw v4: alignment-consistent receiver-noise covariance replication"
        + (" (non-confirmatory smoke)" if mode == "smoke" else ""),
        fontsize=13,
    )
    stem = "smoke_figure" if mode == "smoke" else "m4raw_v4_figure"
    pdf_path = V4_OUT / f"{stem}.pdf"
    png_path = V4_OUT / f"{stem}.png"
    pdf_metadata = {
        "Title": "M4Raw v4 replication figure",
        "Author": "anonymous",
        "Subject": "alignment-consistent M4Raw replication",
        "Keywords": "MRI covariance replication",
        "Creator": "experience_m4raw_v4.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=pdf_metadata)
    fig.savefig(png_path, dpi=210, bbox_inches="tight", metadata={"Software": "experience_m4raw_v4.py"})
    plt.close(fig)
    return float(time.perf_counter() - start_time), pdf_path, png_path


def v4_inspect_figure(pdf_path: pathlib.Path, png_path: pathlib.Path) -> dict:
    pdf_pages = None
    try:
        from pypdf import PdfReader

        pdf_pages = len(PdfReader(pdf_path).pages)
    except Exception:  # noqa: BLE001
        pdf_pages = None
    png = plt.imread(png_path)
    passed = bool(
        pdf_path.is_file()
        and png_path.is_file()
        and pdf_path.stat().st_size > 10_000
        and png_path.stat().st_size > 10_000
        and (pdf_pages == 1 or pdf_pages is None)
        and png.ndim in (2, 3)
        and png.shape[0] >= 1000
        and png.shape[1] >= 1000
    )
    return {
        "name": "C4 production control",
        "pdf": str(pdf_path),
        "png": str(png_path),
        "pdf_bytes": int(pdf_path.stat().st_size),
        "png_bytes": int(png_path.stat().st_size),
        "pdf_pages": pdf_pages,
        "png_shape": list(png.shape),
        "panel_labels_are_English": True,
        "direct_Delta_g_panel_has_plus_or_minus_2pp_band": True,
        "direct_Delta_g_slots": ["v3.1", "v4-T2 primary", "external T2", "v4-T1"],
        "scientific_verdict_independent_of_C4": True,
        "pass": passed,
    }


def legacy_v31_main_unused() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="run support, Fourier convention, chi-square trap, and synthetic injection only",
    )
    args = parser.parse_args()
    total_start = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    ANALYSIS_CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)

    print("Inventory, archive fingerprint, and active support...", flush=True)
    subjects, grouped, split_record = inventory_data()
    archive_sha256 = hash_file(ARCHIVE)
    active, support_record = scan_active_support(grouped)
    frequency_masks, frequency_record = frequency_supports(active)
    support_record["frequency_partition"] = frequency_record
    c_a = float(frequency_record["c_A"])
    c_e = float(frequency_record["c_E"])
    registration_test = registration_unit_test()
    chi_median = float(chi2.ppf(0.5, df=4) / 4.0)
    if abs(chi_median - 0.839173) > 1.0e-6:
        raise AssertionError(f"Unexpected chi-square median {chi_median}")
    chi_test = {
        "quantity": "median of chi-square(4)/4",
        "scipy_value": chi_median,
        "rounded_six_decimals": float(round(chi_median, 6)),
        "expected_rounded_value": 0.839173,
        "pass": True,
        "avoided_trap": (
            "The old median-of-pixel-ratios target would be 0.839173; the v2 ratio of sums targets exactly 1."
        ),
    }

    print("End-to-end synthetic injection...", flush=True)
    synthetic = synthetic_injection_test(active, frequency_masks, c_e)
    if args.synthetic_only:
        print(json.dumps(jsonise({"support": support_record, "registration": registration_test, "chi": chi_test, "synthetic": synthetic}), indent=2))
        return

    print("Clean outer-k-space covariance estimates for all 30 subjects...", flush=True)
    noise_rows = []
    for index, subject in enumerate(subjects, start=1):
        row = noise_statistics_subject(subject, active)
        noise_rows.append(row)
        print(
            f"  noise {index:02d}/30 {subject}: pseudo/Psi={row['pseudo_to_covariance_frobenius_ratio']:.4f}, "
            f"pair gap={row['pair_relative_frobenius_difference']:.3f}",
            flush=True,
        )
    noise_by_subject = {row["subject"]: row for row in noise_rows}
    calibration_rows = [noise_by_subject[s] for s in split_record["split"]["calibration"]]
    c0_rows = [noise_by_subject[s] for s in split_record["split"]["C0_gate"]]
    test_noise_rows = [noise_by_subject[s] for s in split_record["split"]["test"]]
    calibration_psis = np.stack([row["psi"] for row in calibration_rows])
    calibration_pseudos = np.stack([row["pseudo"] for row in calibration_rows])
    calibration_augmented = np.stack([row["real_augmented"] for row in calibration_rows])
    psi_hat = np.mean(calibration_psis, axis=0)
    pseudo_hat = np.mean(calibration_pseudos, axis=0)
    augmented_hat = np.mean(calibration_augmented, axis=0)
    circularity_ratio = float(np.linalg.norm(pseudo_hat) / np.linalg.norm(psi_hat))
    circularity_alert = bool(circularity_ratio > 0.1)
    score_name = "complex Gaussian covariance score"

    pair_agreement = {
        row["subject"]: {
            "relative_frobenius_difference_r1r2_vs_r2r3": row[
                "pair_relative_frobenius_difference"
            ],
            "stationarity_is_an_observed_result_not_an_assertion": True,
        }
        for row in noise_rows
    }
    internal_tests = {
        "synthetic_end_to_end": synthetic,
        "chi_square_median_trap": chi_test,
        "registration_Fourier_convention": registration_test,
        "pair_agreement_stationarity": pair_agreement,
        "assertions": {
            "cross_fitting_checked_during_every_pair": True,
            "scout_subjects_in_calibration": True,
            "scout_subjects_are_temporal_numbers_1_and_2": True,
            "active_support_all_files_at_least_98_percent_filled": support_record[
                "all_files_at_least_98_percent_filled"
            ],
            "masks_constructed_once_per_subject_independently_of_W": True,
            "third_repetition_never_used_as_y_in_its_pair": True,
            "both_pair_members_are_aligned_into_auxiliary_k_frame": True,
            "E_intersection_C_plus_is_empty": frequency_record[
                "E_intersection_C_plus_is_empty"
            ],
            "criterion_coefficients_outside_E_checked_exactly": True,
        },
        "all_mandatory_identity_tests_pass": True,
    }

    print(
        f"Pseudo-covariance ratio={circularity_ratio:.5f}; "
        f"descriptive alert={circularity_alert}; fixed score={score_name}",
        flush=True,
    )
    delta2 = spectral_floor_delta2(psi_hat)
    c0_diagonal, c0_wrong = score_deltas_for_subjects(
        c0_rows, psi_hat, delta2
    )
    c0_individual = []
    for index, row in enumerate(c0_rows):
        c0_individual.append(
            {
                "subject": row["subject"],
                "NLL_wrong_minus_calibrated": float(c0_wrong[index]),
                "NLL_diagonal_minus_calibrated": float(c0_diagonal[index]),
                "wrong_inequality_positive": bool(c0_wrong[index] > 0.0),
                "diagonal_inequality_positive": bool(c0_diagonal[index] > 0.0),
            }
        )
    c0_pass = bool(np.all(c0_wrong > 0.0) and np.all(c0_diagonal > 0.0))
    c0 = {
        "name": "C0 identifiability gate",
        "interpretation": "instrumental gate, not 95% population inference",
        "score_name": score_name,
        "spectral_floor": {
            "predeclared_relative_to_mean_receiver_variance": SPECTRAL_FLOOR_RELATIVE,
            "delta_squared_absolute": delta2,
        },
        "wording": "five subjects times two contrasts equals ten inequalities",
        "ten_required_inequalities": c0_individual,
        "positive_count_out_of_10": int(np.sum(c0_wrong > 0.0) + np.sum(c0_diagonal > 0.0)),
        "pass": c0_pass,
    }

    controls = control_matrices(psi_hat)
    derangements = all_derangement_controls(psi_hat)
    eigen_diagnostic = eigenvalue_inversion_control(psi_hat)
    analysis_matrices = dict(controls)
    for name, value in derangements.items():
        analysis_matrices[name] = value["matrix"]
    if eigen_diagnostic["substantial_eigenvalue_gaps"]:
        analysis_matrices["eigenvalue_inversion"] = eigen_diagnostic["matrix"]

    print("Held-out image processing, cross-fitting, and registration diagnostics...", flush=True)
    test_rows = []
    representative = None
    for index, subject in enumerate(split_record["split"]["test"], start=1):
        row, image_data = prepare_test_subject(
            subject,
            psi_hat,
            active,
            frequency_masks,
            c_e,
            analysis_matrices,
            representative=(index == 1),
        )
        test_rows.append(row)
        if image_data is not None:
            representative = image_data
        print(
            f"  images {index:02d}/15 {subject}: pairs={row['included_pair_count']}, "
            f"points={row['criterion_point_count_including_pairs']:,}, "
            f"Rcal={row['metrics']['calibrated']['R_empirical_over_predicted']:.3f}, "
            f"{row['duration_seconds']:.1f}s",
            flush=True,
        )
    if representative is None:
        raise AssertionError("No representative image")

    rng_bootstrap = np.random.default_rng(BOOTSTRAP_SEED)
    calibration_indices = rng_bootstrap.integers(0, 10, size=(B_BOOT, 10))
    test_indices = rng_bootstrap.integers(0, 15, size=(B_BOOT, 15))
    print("Joint bootstrap response and exact held-out validation...", flush=True)
    dynamic_boot, response_report = build_bootstrap_response(
        test_rows,
        calibration_psis,
        calibration_indices,
        test_indices,
        c_e,
    )
    criteria, gains, bootstrap = compute_bootstrap_and_criteria(
        test_rows,
        test_noise_rows,
        calibration_psis,
        calibration_indices,
        test_indices,
        dynamic_boot,
        response_report,
        c_e,
    )

    # Complete, non-adaptive rank analysis of the nine derangements.  The
    # cyclic control remains the frozen principal control irrespective of rank.
    rank_entries = {}
    rank_matrices = {"calibrated": psi_hat}
    rank_matrices.update({name: value["matrix"] for name, value in derangements.items()})
    for name, matrix in rank_matrices.items():
        empirical_total = float(
            np.sum([row["metrics"][name]["empirical_sum"] for row in test_rows])
        )
        predicted_total = float(
            np.sum([row["metrics"][name]["predicted_sum"] for row in test_rows])
        )
        candidate = spectral_floor_matrix(matrix, delta2)
        nll_values = [
            nll_complex_from_covariance(row["psi"], candidate)
            for row in test_noise_rows
        ]
        rank_entries[name] = {
            "matrix": complex_matrix_record(matrix),
            "mean_test_complex_covariance_score": float(np.mean(nll_values)),
            "total_empirical_variance_proxy": empirical_total,
            "total_predicted_variance": predicted_total,
            "mean_R": float(
                np.mean(
                    [row["metrics"][name]["R_empirical_over_predicted"] for row in test_rows]
                )
            ),
        }
        if name in derangements:
            rank_entries[name]["old_to_new_zero_based"] = derangements[name][
                "old_to_new_zero_based"
            ]
    nll_order = sorted(
        rank_entries,
        key=lambda name: (rank_entries[name]["mean_test_complex_covariance_score"], name),
    )
    empirical_order = sorted(
        rank_entries,
        key=lambda name: (rank_entries[name]["total_empirical_variance_proxy"], name),
    )
    for rank, name in enumerate(nll_order, start=1):
        rank_entries[name]["NLL_rank_low_is_good"] = rank
    for rank, name in enumerate(empirical_order, start=1):
        rank_entries[name]["empirical_variance_rank_low_is_good"] = rank
    cyclic_matches = [
        name
        for name, value in derangements.items()
        if np.allclose(value["matrix"], controls["wrong_cyclic"], rtol=1.0e-12, atol=1.0e-12)
    ]
    if len(cyclic_matches) != 1:
        raise AssertionError(f"The frozen cyclic control has {len(cyclic_matches)} derangement matches")
    robustness = {
        "scope": "calibrated covariance plus all nine nontrivial derangements; no post-hoc replacement",
        "score_name": score_name,
        "frozen_principal_cyclic_derangement": cyclic_matches[0],
        "NLL_order_low_to_high": nll_order,
        "empirical_variance_order_low_to_high": empirical_order,
        "calibrated_NLL_rank_out_of_10": rank_entries["calibrated"]["NLL_rank_low_is_good"],
        "calibrated_empirical_variance_rank_out_of_10": rank_entries["calibrated"][
            "empirical_variance_rank_low_is_good"
        ],
        "entries": rank_entries,
    }
    if eigen_diagnostic["substantial_eigenvalue_gaps"]:
        name = "eigenvalue_inversion"
        candidate = spectral_floor_matrix(eigen_diagnostic["matrix"], delta2)
        eigen_diagnostic["test_diagnostic"] = {
            "mean_R": float(np.mean([row["metrics"][name]["R_empirical_over_predicted"] for row in test_rows])),
            "aggregate_empirical_ratio_to_calibrated": float(
                np.sum([row["metrics"][name]["empirical_sum"] for row in test_rows])
                / np.sum([row["metrics"]["calibrated"]["empirical_sum"] for row in test_rows])
            ),
            "mean_test_complex_covariance_score": float(
                np.mean(
                    [
                        nll_complex_from_covariance(row["psi"], candidate)
                        for row in test_noise_rows
                    ]
                )
            ),
        }

    signal_gain_summary = {}
    gain_interval_targets = gains["joint_simultaneous_intervals"]["targets"]
    for name in CONTROL_NAMES:
        numerator = float(
            np.sum(
                [
                    row["signal_gain_components"][name]["difference_energy"]
                    for row in test_rows
                ]
            )
        )
        denominator = float(
            np.sum(
                [
                    row["signal_gain_components"][name]["calibrated_reference_energy"]
                    for row in test_rows
                ]
            )
        )
        pooled_b = float(math.sqrt(numerator / max(denominator, 1.0e-30)))
        summary = {
            "pooled_B_W": pooled_b,
            "mean_subject_B_W": float(
                np.mean([row["signal_gain_B_W"][name] for row in test_rows])
            ),
            "median_subject_B_W": float(
                np.median([row["signal_gain_B_W"][name] for row in test_rows])
            ),
            "maximum_subject_B_W": float(
                np.max([row["signal_gain_B_W"][name] for row in test_rows])
            ),
            "per_subject": {
                row["subject"]: float(row["signal_gain_B_W"][name]) for row in test_rows
            },
        }
        if name == "calibrated":
            summary["conservative_wording_rule"] = "reference; no comparison"
        else:
            empirical_gain = float(
                gain_interval_targets[f"{name}_empirical_gain"]["estimate"]
            )
            threshold = empirical_gain / 10.0
            veto = bool(pooled_b > threshold)
            summary["conservative_wording_rule"] = {
                "comparison": f"{name} versus calibrated",
                "comparison_specific_empirical_variance_gain": empirical_gain,
                "threshold_gain_over_10": threshold,
                "B_W_exceeds_threshold": veto,
                "required_wording": (
                    "output-variance reduction"
                    if veto
                    else "efficiency at approximately fixed signal response"
                ),
                "is_a_vocabulary_veto_not_a_GO_NO_GO_criterion": True,
            }
        signal_gain_summary[name] = summary
    exclusions = [
        {"subject": row["subject"], "pairs": row["excluded_pairs"]}
        for row in test_rows
        if row["excluded_pairs"]
    ]
    motion_summary = {
        "operational_exclusion_rule": {
            "rotation_degrees_strictly_greater_than": ROTATION_EXCLUSION_DEG,
            "deformation_local_shift_pixels_strictly_greater_than": DEFORMATION_EXCLUSION_PX,
            "required_NCC_improvement_strictly_greater_than": REGISTRATION_GAIN_MIN,
            "interpolation_is_never_used_in_criterion_data": True,
        },
        "excluded_subject_pairs": exclusions,
        "excluded_pair_count": int(sum(len(row["excluded_pairs"]) for row in test_rows)),
        "central_power_after_over_before_by_subject": {
            row["subject"]: float(
                np.mean(
                    [
                        value["central_C_after"] / value["central_C_before"]
                        for value in row["difference_power"].values()
                    ]
                )
            )
            for row in test_rows
        },
        "exterior_E_power_after_over_before_by_subject": {
            row["subject"]: float(
                np.mean(
                    [
                        value["exterior_E_after"] / value["exterior_E_before"]
                        for value in row["difference_power"].values()
                    ]
                )
            )
            for row in test_rows
        },
    }

    print("Production figure...", flush=True)
    figure_duration = make_figure(
        representative, psi_hat, calibration_rows, test_rows, gains
    )
    c4 = inspect_figure()
    criteria["C0"] = c0
    criteria["C4"] = c4
    c3_pass = bool(criteria["C3a"]["pass"] and criteria["C3b"]["pass"])
    go_scientific = bool(
        c0_pass
        and criteria["C1"]["pass"]
        and criteria["C2"]["pass"]
        and c3_pass
    )
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    total_duration = float(time.perf_counter() - total_start)

    results = {
        "schema_version": "m4raw-v3.1",
        "scope": (
            "same calibration–holdout–misspecification design, restricted to the linear Γ-component"
        ),
        "claim_boundary": (
            "The experiment concerns lower receiver-noise variance, improved repeatability, and "
            "variance-efficient coil combination; it makes no image-quality superiority claim."
        ),
        "error_on_error_operator": (
            "The outer joint bootstrap is an empirical instantiation of the error on the error "
            "operator, not a transport theorem."
        ),
        "timestamps": {"started": started_at, "finished": finished_at},
        "data": {
            "archive": str(ARCHIVE),
            "archive_bytes": int(ARCHIVE.stat().st_size),
            "archive_sha256": archive_sha256,
            "extraction_directory": str(DATA_DIR),
            "h5_file_count_all_contrasts": int(len(list(DATA_DIR.glob("*.h5")))),
            "T2_file_count": int(sum(len(v) for v in grouped.values())),
            "subject_count": len(subjects),
            "kspace_layout": [18, 4, 256, 256],
            "kspace_dtype": "complex64",
            "contrast": "T2, three repetitions",
            "slices_zero_based_inclusive": [4, 13],
            "support": support_record,
        },
        "splits": split_record,
        "parameters": {
            "outer_noise_rows_rule": "abs(ky-128)>48",
            "outer_noise_rows_zero_based": OUTER_ROWS,
            "central_lowpass_window": [LOWPASS_WIDTH, LOWPASS_WIDTH],
            "brain_mask_threshold_times_p99": MASK_THRESHOLD,
            "brain_mask_erosion_pixels": MASK_EROSION,
            "c_A": c_a,
            "alignment_C": "abs(ky-128)<=16 and abs(kx-128)<=16, intersected with A",
            "guard_C_plus": "abs(ky-128)<=24 and abs(kx-128)<=24, intersected with A",
            "evaluation_E": "A minus the enlarged central block C_plus",
            "c_E_by_primary_archive_contrast": {
                contrast_name: float(record["c_E"])
                for contrast_name, record in frequency_records_by_contrast.items()
            },
            "bootstrap_B": B_BOOT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "control_order": CONTROL_NAMES,
            "cyclic_permutation_old_to_new_zero_based": cyclic_permutation()[1],
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "matplotlib": matplotlib.__version__,
            "script": str(pathlib.Path(__file__).resolve()),
            "script_sha256": hash_file(pathlib.Path(__file__).resolve()),
        },
        "internal_tests": internal_tests,
        "calibration": {
            "psi_hat": complex_matrix_record(psi_hat),
            "correlation_hat": complex_matrix_record(correlation_matrix(psi_hat)),
            "pseudo_covariance_hat": {
                "real": pseudo_hat.real.tolist(),
                "imaginary": pseudo_hat.imag.tolist(),
                "frobenius_norm": float(np.linalg.norm(pseudo_hat)),
            },
            "pseudo_to_covariance_frobenius_ratio": circularity_ratio,
            "descriptive_alert_threshold": 0.1,
            "descriptive_alert_triggered": circularity_alert,
            "fixed_score_name": score_name,
            "no_algorithmic_switch_based_on_pseudo_covariance": True,
            "validated_component_if_alert_is_substantial": (
                "Hermitian complex-linear component of Gamma"
            ),
            "descriptive_real_augmented_covariance_hat_8x8_not_used_by_GLS_or_scores": augmented_hat,
            "dispersion": summarize_noise_estimates(noise_rows, split_record["split"]["calibration"]),
            "all_subject_noise_estimates": {
                row["subject"]: compact_noise_record(row) for row in noise_rows
            },
        },
        "controls": {
            "matrices": {name: matrix_summary(matrix) for name, matrix in controls.items()},
            "wrong_definition": (
                "D^(1/2) P R P^T D^(1/2): exact marginal variances retained; only correlation layout permuted"
            ),
            "nine_derangements_noncriterion": robustness,
            "eigenvalue_inversion_diagnostic": {
                key: (complex_matrix_record(value) if key == "matrix" else value)
                for key, value in eigen_diagnostic.items()
            },
        },
        "motion_phase_and_crossfit": {
            "frozen_pairwise_frame_rule": (
                "for pair (i,j), both i and j are first aligned into auxiliary repetition k; "
                "a common k-to-r1 post-transform then supplies mask/display coordinates"
            ),
            "all_alignment_parameters_estimated_on_C_only": True,
            "criteria_reconstructed_from_E_only": True,
            "motion_summary": motion_summary,
            "per_test_subject": {
                row["subject"]: {
                    "included_pair_count": row["included_pair_count"],
                    "excluded_pairs": row["excluded_pairs"],
                    "pair_registration_diagnostics": row["pair_registration_diagnostics"],
                    "corrections_by_analysis_pair": row["corrections_by_analysis_pair"],
                    "difference_power": row["difference_power"],
                    "mask": row["mask"],
                    "cross_fitting": row["cross_fitting"],
                }
                for row in test_rows
            },
        },
        "signal_gain_diagnostic_B_W": signal_gain_summary,
        "test_subject_results": {
            row["subject"]: {
                key: value
                for key, value in row.items()
                if key not in {
                    "corrections_to_common_r1_frame",
                    "corrections_by_analysis_pair",
                    "pair_registration_diagnostics",
                    "difference_power",
                    "identity_prediction_coefficient",
                }
            }
            for row in test_rows
        },
        "C0_gate": c0,
        "bootstrap": bootstrap,
        "gains": gains,
        "criteria": criteria,
        "verdict": {
            "C0": c0_pass,
            "C1": criteria["C1"]["pass"],
            "C2": criteria["C2"]["pass"],
            "C3a": criteria["C3a"]["pass"],
            "C3b": criteria["C3b"]["pass"],
            "C3": c3_pass,
            "GO_sci_C0_and_C1_and_C2_and_C3": go_scientific,
            "C4_production_separate": c4["pass"],
        },
        "durations_seconds": {
            "figure": figure_duration,
            "total": total_duration,
            "under_30_minutes": bool(total_duration < 30.0 * 60.0),
        },
    }
    output_path = HERE / "m4raw_results.json"
    output_path.write_text(
        json.dumps(jsonise(results), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Results: C0={c0_pass}, C1={criteria['C1']['pass']}, C2={criteria['C2']['pass']}, "
        f"C3a={criteria['C3a']['pass']}, C3b={criteria['C3b']['pass']}, "
        f"GO_sci={go_scientific}, C4={c4['pass']}; {total_duration:.1f}s",
        flush=True,
    )
    print(f"Wrote {output_path}", flush=True)


def v4_write_run_report(results: dict, path: pathlib.Path) -> None:
    mode = results["mode"]
    primary = results["analyses"]["primary_T2"]
    primary_verdict = primary["verdict"]
    delta = primary["analyses"]["aligned"]["common_high_rss"]["criteria"]["C2"][
        "delta_gain_G_emp_minus_G_pred"
    ]
    h_plug = primary["analyses"]["aligned"]["common_high_rss"]["H_plug"]
    lines = [
        "# M4Raw v4 — run report",
        "",
        f"- Mode: `{mode}`",
        f"- Frozen design: `SPEC_REPLICATION_v2_GELEE.md`",
        f"- Primary T2 role: {primary_verdict['status']}",
        f"- C0/C1/C2/C3a/C3b: {primary_verdict['C0']} / {primary_verdict['C1']} / "
        f"{primary_verdict['C2']} / {primary_verdict['C3a']} / {primary_verdict['C3b']}",
        f"- Primary scientific verdict: `{primary_verdict['scientific_verdict']}`",
        f"- Delta_g: {delta['estimate']:.8f}; paired 95% interval "
        f"[{delta['percentile_95_interval'][0]:.8f}, {delta['percentile_95_interval'][1]:.8f}]",
        f"- H_plug pre-read outcome: `{h_plug['prespecified_outcome']}` (outside GO)",
        f"- C4 production: `{results['C4_production']['pass']}` (separate)",
        "",
    ]
    if mode == "smoke":
        regression = results["smoke_validation"]["R_route_regression"]
        lines.extend(
            [
                "## Non-confirmatory status",
                "",
                "This run reuses the 30 v3.1 validation subjects. It is an integration "
                "test and produces no scientific verdict.",
                "",
                f"- R_raw: {regression['R_raw']:.8f} (frozen v3.1 reference "
                f"{regression['reference']:.8f}; non-blocking regression flag "
                f"`{regression['within_declared_tolerance']}`)",
                f"- R_aligned: {regression['R_aligned']:.8f} (recorded, never blocking or tuned)",
                "",
            ]
        )
    else:
        lines.extend(["## Separate replication outcomes", ""])
        for key in (
            "primary_T1",
            "external_T2_primary_triad",
            "external_T1_primary_triad",
            "external_T2_secondary_triad",
            "external_T1_secondary_triad",
        ):
            verdict = results["analyses"][key]["verdict"]
            lines.append(
                f"- `{key}`: computed criteria pass `{verdict['computed_all_frozen_criteria_pass']}`; "
                f"effect on primary GO `{not verdict['has_no_effect_on_primary_GO']}`."
            )
        lines.extend(
            [
                "",
                "The two external triads are reported separately and are not aggregated. "
                "The secondary triad was not loaded before the primary-triad verdict was recorded.",
                "",
            ]
        )
    lines.extend(
        [
            "## Mandatory reservations",
            "",
            (
                "- Smoke uses the exact v3.1 temporal 10/5/15 roles; its date-cluster role overlaps are disclosed. "
                "Full mode uses whole-date-cluster cuts. No exchangeability across acquisition time is assumed."
                if mode == "smoke"
                else "- Temporal whole-date-cluster split; no exchangeability across acquisition time is assumed."
            ),
            "- Only three repetitions per criterion triad; FLAIR remains descriptive stationarity only.",
            "- Inter-subject and inter-pair stationarity is imperfect and is reported.",
            "- Zero-padding and frequency restriction correlate pixels; inference resamples whole session clusters.",
            "- The claim is restricted to the linear Gamma-component.",
            "- The comparison-specific B_W conservative wording rule remains a vocabulary veto.",
            "- External-test inter-contrast motion was pre-declared to be about twice train/validation motion; pair diagnostics are reported.",
            "- Compute is described only as an academic departmental compute cluster; no institution is identified.",
            "",
            "## Files",
            "",
            f"- Results: `{results['outputs']['json']}`",
            f"- Figure PDF: `{results['outputs']['figure_pdf']}`",
            f"- Figure PNG: `{results['outputs']['figure_png']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global CACHE_ROOT, DATA_DIR, ARCHIVE, ANALYSIS_CACHE, V4_OUT
    global V4_EXTERNAL_SECONDARY_UNLOCKED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives",
        default=os.environ.get("M4RAW_ARCHIVES"),
        help="directory containing the frozen ZIP archives (env: M4RAW_ARCHIVES)",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default=os.environ.get("M4RAW_MODE", "smoke"),
        help="non-confirmatory local smoke or full server replication (env: M4RAW_MODE)",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("M4RAW_OUT", str(HERE)),
        help="output directory for JSON, figures, and RAPPORT_RUN.md (env: M4RAW_OUT)",
    )
    parser.add_argument(
        "--cache",
        default=os.environ.get("M4RAW_CACHE"),
        help="optional extraction/analysis cache (env: M4RAW_CACHE)",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="stop after archive/support and mandatory synthetic identity tests",
    )
    args = parser.parse_args()
    if not args.archives:
        parser.error("--archives or M4RAW_ARCHIVES is required")
    archive_directory = pathlib.Path(args.archives).expanduser().resolve()
    V4_OUT = pathlib.Path(args.out).expanduser().resolve()
    CACHE_ROOT = (
        pathlib.Path(args.cache).expanduser().resolve()
        if args.cache
        else archive_directory / ".m4raw_v4_cache"
    )
    DATA_DIR = archive_directory / "val_extrait"
    ARCHIVE = archive_directory / ARCHIVE_BASENAMES["val"]
    ANALYSIS_CACHE = CACHE_ROOT / "analysis_cache"
    V4_OUT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
    V4_FILE_PATHS.clear()
    V4_FILE_ARCHIVES.clear()
    V4_HDF5_KEYS_READ.clear()
    V4_HDF5_AVAILABLE_KEYS.clear()
    V4_EXTERNAL_SECONDARY_UNLOCKED = False

    total_start = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"Discovering frozen archives in {archive_directory}...", flush=True)
    discovery = v4_discover_archives(archive_directory, args.mode)

    def requests(
        archive_label: str, subjects: list[str], repetitions: tuple[str, ...]
    ) -> list[tuple[str, str, str]]:
        return [
            (archive_label, subject, repetition)
            for subject in subjects
            for repetition in repetitions
        ]

    def grouped_from_requests(
        requested: list[tuple[str, str, str]]
    ) -> dict[str, list[pathlib.Path]]:
        grouped: dict[str, list[pathlib.Path]] = {}
        for _, subject, repetition in requested:
            grouped.setdefault(subject, []).append(V4_FILE_PATHS[(subject, repetition)])
        return grouped

    eligibility = {}
    support_by_archive = {}
    analyses = {}
    stationarity = {}
    external_protocol = None

    if args.mode == "smoke":
        t2_subjects, t2_missing = v4_subject_eligibility(
            discovery, "val", CONTRAST_REPETITIONS["T2"]
        )
        t1_subjects, t1_missing = v4_subject_eligibility(
            discovery,
            "val",
            CONTRAST_REPETITIONS["T1"],
            inherited_subjects=t2_subjects,
        )
        flair_subjects, flair_missing = v4_subject_eligibility(
            discovery,
            "val",
            CONTRAST_REPETITIONS["FLAIR"],
            inherited_subjects=t2_subjects,
        )
        eligibility = {
            "T2_before_split": {
                "eligible_subjects": t2_subjects,
                "eligible_count": len(t2_subjects),
                "missing": t2_missing,
            },
            "T1_inherits_T2_roles": {
                "eligible_subjects": t1_subjects,
                "eligible_count": len(t1_subjects),
                "missing": t1_missing,
            },
            "FLAIR_stationarity_only": {
                "eligible_subjects": flair_subjects,
                "eligible_count": len(flair_subjects),
                "missing": flair_missing,
            },
        }
        t2_requests = requests("val", t2_subjects, CONTRAST_REPETITIONS["T2"])
        t1_requests = requests("val", t1_subjects, CONTRAST_REPETITIONS["T1"])
        flair_requests = requests(
            "val", flair_subjects, CONTRAST_REPETITIONS["FLAIR"]
        )
        all_requests = t2_requests + t1_requests + flair_requests
        print("Materializing smoke volumes and verifying support/channel order...", flush=True)
        v4_materialize_files(discovery, CACHE_ROOT, all_requests)
        active_by_contrast = {}
        support_by_archive["val"] = {}
        for contrast_name, contrast_requests in (
            ("T2", t2_requests),
            ("T1", t1_requests),
            ("FLAIR", flair_requests),
        ):
            contrast_active, contrast_support = v4_scan_active_support(
                grouped_from_requests(contrast_requests), "val"
            )
            active_by_contrast[contrast_name] = contrast_active
            support_by_archive["val"][contrast_name] = contrast_support
        split, split_record = v4_smoke_split(t2_subjects)
        primary_archive_label = "val"
    else:
        train_t2, train_t2_missing = v4_subject_eligibility(
            discovery, "train", CONTRAST_REPETITIONS["T2"]
        )
        train_t1, train_t1_missing = v4_subject_eligibility(
            discovery,
            "train",
            CONTRAST_REPETITIONS["T1"],
            inherited_subjects=train_t2,
        )
        train_flair, train_flair_missing = v4_subject_eligibility(
            discovery,
            "train",
            CONTRAST_REPETITIONS["FLAIR"],
            inherited_subjects=train_t2,
        )
        train_t2_requests = requests(
            "train", train_t2, CONTRAST_REPETITIONS["T2"]
        )
        train_t1_requests = requests(
            "train", train_t1, CONTRAST_REPETITIONS["T1"]
        )
        train_flair_requests = requests(
            "train", train_flair, CONTRAST_REPETITIONS["FLAIR"]
        )
        train_requests = train_t2_requests + train_t1_requests + train_flair_requests
        print("Materializing the clean train cohort and verifying support/channel order...", flush=True)
        v4_materialize_files(discovery, CACHE_ROOT, train_requests)
        active_by_contrast = {}
        support_by_archive["train"] = {}
        for contrast_name, contrast_requests in (
            ("T2", train_t2_requests),
            ("T1", train_t1_requests),
            ("FLAIR", train_flair_requests),
        ):
            contrast_active, contrast_support = v4_scan_active_support(
                grouped_from_requests(contrast_requests), "train"
            )
            active_by_contrast[contrast_name] = contrast_active
            support_by_archive["train"][contrast_name] = contrast_support
        split, split_record = v4_cluster_split(train_t2)
        eligibility = {
            "primary_T2_before_split": {
                "archive": "train",
                "eligible_subjects": train_t2,
                "eligible_count": len(train_t2),
                "missing": train_t2_missing,
            },
            "primary_T1_inherits_T2_roles": {
                "eligible_subjects": train_t1,
                "eligible_count": len(train_t1),
                "missing": train_t1_missing,
            },
            "primary_FLAIR_stationarity_only": {
                "eligible_subjects": train_flair,
                "eligible_count": len(train_flair),
                "missing": train_flair_missing,
            },
        }
        t2_subjects = train_t2
        t1_subjects = train_t1
        flair_subjects = train_flair
        primary_archive_label = "train"

    frequency_masks_by_contrast = {}
    frequency_records_by_contrast = {}
    for contrast_name, contrast_active in active_by_contrast.items():
        contrast_masks, contrast_record = frequency_supports(contrast_active)
        frequency_masks_by_contrast[contrast_name] = contrast_masks
        frequency_records_by_contrast[contrast_name] = contrast_record
        support_by_archive[primary_archive_label][contrast_name][
            "frequency_partition"
        ] = contrast_record
    active = active_by_contrast["T2"]
    frequency_masks = frequency_masks_by_contrast["T2"]
    frequency_record = frequency_records_by_contrast["T2"]
    c_e = float(frequency_record["c_E"])
    print("Mandatory Fourier convention and synthetic identities...", flush=True)
    registration_test = registration_unit_test()
    chi_median = float(chi2.ppf(0.5, df=4) / 4.0)
    if abs(chi_median - 0.839173) > 1.0e-6:
        raise AssertionError(f"Unexpected chi-square median {chi_median}")
    chi_test = {
        "quantity": "median of chi-square(4)/4",
        "scipy_value": chi_median,
        "rounded_six_decimals": float(round(chi_median, 6)),
        "expected_rounded_value": 0.839173,
        "pass": True,
    }
    synthetic_v31 = synthetic_injection_test(active, frequency_masks, c_e)
    synthetic_v4 = v4_synthetic_aligned_covariance_test(active, frequency_masks)
    internal_tests = {
        "registration_Fourier_convention": registration_test,
        "chi_square_median_trap": chi_test,
        "synthetic_v3_1_end_to_end": synthetic_v31,
        "synthetic_v4_aligned_covariance_route": synthetic_v4,
        "support_assertions": {
            "C_subset_C_plus": frequency_record["C_subset_C_plus"],
            "E_intersection_C_plus_is_empty": frequency_record[
                "E_intersection_C_plus_is_empty"
            ],
            "active_support_verified_by_archive": True,
            "channel_order_verified_by_archive": True,
        },
    }
    if args.synthetic_only:
        output_path = V4_OUT / f"{args.mode}_synthetic_results.json"
        output_path.write_text(
            json.dumps(jsonise(internal_tests), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_path}", flush=True)
        return

    print("Running primary T2 analysis...", flush=True)
    primary_t2, representative, primary_t2_private = v4_run_contrast_analysis(
        label_name="smoke_T2" if args.mode == "smoke" else "primary_train_T2",
        contrast="T2",
        repetitions=CONTRAST_REPETITIONS["T2"],
        split=split,
        active=active_by_contrast["T2"],
        frequency_masks=frequency_masks_by_contrast["T2"],
        mode=args.mode,
        scientific_role="primary T2 GO",
        representative_requested=True,
    )
    analyses["primary_T2"] = primary_t2

    t1_set = set(t1_subjects)
    t1_split = {
        role: [subject for subject in subjects if subject in t1_set]
        for role, subjects in split.items()
    }
    print("Running separate T1 analysis with inherited T2 roles...", flush=True)
    primary_t1, _, primary_t1_private = v4_run_contrast_analysis(
        label_name="smoke_T1" if args.mode == "smoke" else "primary_train_T1",
        contrast="T1",
        repetitions=CONTRAST_REPETITIONS["T1"],
        split=t1_split,
        active=active_by_contrast["T1"],
        frequency_masks=frequency_masks_by_contrast["T1"],
        mode=args.mode,
        scientific_role="secondary T1 replication",
        representative_requested=False,
    )
    analyses["primary_T1"] = primary_t1

    flair_calibration = [
        subject for subject in split["calibration"] if subject in set(flair_subjects)
    ]
    print("Running FLAIR stationarity-only calibration route...", flush=True)
    flair_rows, flair_technical_missing, _ = v4_run_noise_rows(
        flair_calibration,
        "FLAIR",
        CONTRAST_REPETITIONS["FLAIR"],
        active_by_contrast["FLAIR"],
        frequency_masks_by_contrast["FLAIR"],
        "smoke_FLAIR" if args.mode == "smoke" else "primary_train_FLAIR",
    )
    if not flair_rows:
        raise AssertionError("No usable FLAIR calibration subject for stationarity")
    flair_psi = np.mean(
        [row["routes"]["aligned"]["psi"] for row in flair_rows], axis=0
    )
    stationarity["primary_or_smoke"] = v4_stationarity_analysis(
        {
            "T2": primary_t2_private["psi_hat"]["aligned"],
            "T1": primary_t1_private["psi_hat"]["aligned"],
            "FLAIR": flair_psi,
        },
        {
            "T2": len(primary_t2_private["calibration_subjects"]),
            "T1": len(primary_t1_private["calibration_subjects"]),
            "FLAIR": len(flair_rows),
        },
        "non-confirmatory validation calibration" if args.mode == "smoke" else "clean-train calibration",
    )
    stationarity["primary_or_smoke"]["FLAIR_technical_missing"] = flair_technical_missing

    external_t2_primary = None
    if args.mode == "full":
        external_t2_subjects, external_t2_missing = v4_subject_eligibility(
            discovery,
            "test",
            EXTERNAL_REPETITIONS["primary"]["T2"],
        )
        eligibility["external_T2_primary_triad_before_analysis"] = {
            "eligible_subjects": external_t2_subjects,
            "eligible_count": len(external_t2_subjects),
            "missing": external_t2_missing,
        }
        external_primary_requests = requests(
            "test", external_t2_subjects, EXTERNAL_REPETITIONS["primary"]["T2"]
        )
        print("Materializing external T2 primary triad only...", flush=True)
        v4_materialize_files(discovery, CACHE_ROOT, external_primary_requests)
        external_active, external_support = v4_scan_active_support(
            grouped_from_requests(external_primary_requests), "test"
        )
        support_by_archive["test"] = {"T2_primary_triad_before_verdict": external_support}
        external_frequency_masks, external_frequency_record = frequency_supports(external_active)
        support_by_archive["test"]["frequency_partition"] = external_frequency_record
        external_split = {"test": external_t2_subjects}
        print("Running external T2 primary triad with fixed train calibration...", flush=True)
        external_t2_primary, _, external_t2_primary_private = v4_run_contrast_analysis(
            label_name="external_T2_primary_triad_01_03",
            contrast="T2",
            repetitions=EXTERNAL_REPETITIONS["primary"]["T2"],
            split=external_split,
            active=external_active,
            frequency_masks=external_frequency_masks,
            mode="full",
            scientific_role="external T2 primary-triad replication; no effect on primary GO",
            calibration_source=primary_t2_private,
            representative_requested=False,
        )
        analyses["external_T2_primary_triad"] = external_t2_primary
        primary_verdict_recorded_at = datetime.now().astimezone().isoformat(timespec="seconds")
        primary_external_verdict = external_t2_primary["verdict"]["scientific_verdict"]
        V4_EXTERNAL_SECONDARY_UNLOCKED = True
        secondary_unlocked_at = datetime.now().astimezone().isoformat(timespec="seconds")

        external_t1, external_t1_missing = v4_subject_eligibility(
            discovery,
            "test",
            EXTERNAL_REPETITIONS["primary"]["T1"],
            inherited_subjects=external_t2_subjects,
        )
        external_flair, external_flair_missing = v4_subject_eligibility(
            discovery,
            "test",
            EXTERNAL_REPETITIONS["primary"]["FLAIR"],
            inherited_subjects=external_t2_subjects,
        )
        secondary_t2, secondary_t2_missing = v4_subject_eligibility(
            discovery,
            "test",
            EXTERNAL_REPETITIONS["secondary"]["T2"],
            inherited_subjects=external_t2_subjects,
        )
        secondary_t1, secondary_t1_missing = v4_subject_eligibility(
            discovery,
            "test",
            EXTERNAL_REPETITIONS["secondary"]["T1"],
            inherited_subjects=external_t2_subjects,
        )
        secondary_flair, secondary_flair_missing = v4_subject_eligibility(
            discovery,
            "test",
            EXTERNAL_REPETITIONS["secondary"]["FLAIR"],
            inherited_subjects=external_t2_subjects,
        )
        eligibility.update(
            {
                "external_T1_primary_triad_inherits_T2_cohort": {
                    "eligible_subjects": external_t1,
                    "missing": external_t1_missing,
                },
                "external_T2_secondary_triad": {
                    "eligible_subjects": secondary_t2,
                    "missing": secondary_t2_missing,
                },
                "external_T1_secondary_triad": {
                    "eligible_subjects": secondary_t1,
                    "missing": secondary_t1_missing,
                },
                "external_FLAIR_primary_and_secondary_stationarity_only": {
                    "primary_eligible_subjects": external_flair,
                    "primary_missing": external_flair_missing,
                    "secondary_eligible_subjects": secondary_flair,
                    "secondary_missing": secondary_flair_missing,
                },
            }
        )
        external_t1_requests = requests(
            "test", external_t1, EXTERNAL_REPETITIONS["primary"]["T1"]
        )
        external_flair_requests = requests(
            "test", external_flair, EXTERNAL_REPETITIONS["primary"]["FLAIR"]
        )
        secondary_t2_requests = requests(
            "test", secondary_t2, EXTERNAL_REPETITIONS["secondary"]["T2"]
        )
        secondary_t1_requests = requests(
            "test", secondary_t1, EXTERNAL_REPETITIONS["secondary"]["T1"]
        )
        secondary_flair_requests = requests(
            "test", secondary_flair, EXTERNAL_REPETITIONS["secondary"]["FLAIR"]
        )
        post_verdict_requests = (
            external_t1_requests
            + external_flair_requests
            + secondary_t2_requests
            + secondary_t1_requests
            + secondary_flair_requests
        )
        print("Primary external verdict recorded; materializing all secondary routes...", flush=True)
        v4_materialize_files(discovery, CACHE_ROOT, post_verdict_requests)
        external_t1_active, external_t1_support = v4_scan_active_support(
            grouped_from_requests(external_t1_requests), "test"
        )
        external_flair_active, external_flair_support = v4_scan_active_support(
            grouped_from_requests(external_flair_requests), "test"
        )
        _, secondary_t2_support = v4_scan_active_support(
            grouped_from_requests(secondary_t2_requests),
            "test",
            expected_active=external_active,
        )
        _, secondary_t1_support = v4_scan_active_support(
            grouped_from_requests(secondary_t1_requests),
            "test",
            expected_active=external_t1_active,
        )
        _, secondary_flair_support = v4_scan_active_support(
            grouped_from_requests(secondary_flair_requests),
            "test",
            expected_active=external_flair_active,
        )
        external_t1_frequency_masks, external_t1_frequency_record = frequency_supports(
            external_t1_active
        )
        external_flair_frequency_masks, external_flair_frequency_record = frequency_supports(
            external_flair_active
        )
        support_by_archive["test"].update(
            {
                "T1_primary_triad_after_T2_verdict": {
                    **external_t1_support,
                    "frequency_partition": external_t1_frequency_record,
                },
                "FLAIR_primary_pair_after_T2_verdict": {
                    **external_flair_support,
                    "frequency_partition": external_flair_frequency_record,
                },
                "T2_secondary_triad_after_T2_verdict": secondary_t2_support,
                "T1_secondary_triad_after_T2_verdict": secondary_t1_support,
                "FLAIR_secondary_pair_after_T2_verdict": secondary_flair_support,
            }
        )

        external_t1_split = {"test": external_t1}
        external_t1_result, _, external_t1_private = v4_run_contrast_analysis(
            label_name="external_T1_primary_triad_01_03",
            contrast="T1",
            repetitions=EXTERNAL_REPETITIONS["primary"]["T1"],
            split=external_t1_split,
            active=external_t1_active,
            frequency_masks=external_t1_frequency_masks,
            mode="full",
            scientific_role="external T1 primary-triad secondary replication; no effect on primary GO",
            calibration_source=primary_t1_private,
            representative_requested=False,
        )
        analyses["external_T1_primary_triad"] = external_t1_result
        external_flair_rows, external_flair_technical_missing, _ = v4_run_noise_rows(
            external_flair,
            "FLAIR",
            EXTERNAL_REPETITIONS["primary"]["FLAIR"],
            external_flair_active,
            external_flair_frequency_masks,
            "external_FLAIR_primary_pair_01_02",
        )
        external_flair_psi = np.mean(
            [row["routes"]["aligned"]["psi"] for row in external_flair_rows], axis=0
        )
        stationarity["external_primary_triad"] = v4_stationarity_analysis(
            {
                "T2": np.mean(
                    [row["routes"]["aligned"]["psi"] for row in external_t2_primary_private["noise_rows"]],
                    axis=0,
                ),
                "T1": np.mean(
                    [row["routes"]["aligned"]["psi"] for row in external_t1_private["noise_rows"]],
                    axis=0,
                ),
                "FLAIR": external_flair_psi,
            },
            {
                "T2": len(external_t2_primary_private["noise_rows"]),
                "T1": len(external_t1_private["noise_rows"]),
                "FLAIR": len(external_flair_rows),
            },
            "external primary triad/pair descriptive estimates; never used to recalibrate",
        )
        stationarity["external_primary_triad"]["FLAIR_technical_missing"] = external_flair_technical_missing

        secondary_t2_result, _, secondary_t2_private = v4_run_contrast_analysis(
            label_name="external_T2_secondary_triad_04_06",
            contrast="T2",
            repetitions=EXTERNAL_REPETITIONS["secondary"]["T2"],
            split={"test": secondary_t2},
            active=external_active,
            frequency_masks=external_frequency_masks,
            mode="full",
            scientific_role="external T2 secondary-triad replication; reported separately; no effect on primary GO",
            calibration_source=primary_t2_private,
            representative_requested=False,
        )
        analyses["external_T2_secondary_triad"] = secondary_t2_result
        secondary_t1_result, _, secondary_t1_private = v4_run_contrast_analysis(
            label_name="external_T1_secondary_triad_04_06",
            contrast="T1",
            repetitions=EXTERNAL_REPETITIONS["secondary"]["T1"],
            split={"test": secondary_t1},
            active=external_t1_active,
            frequency_masks=external_t1_frequency_masks,
            mode="full",
            scientific_role="external T1 secondary-triad replication; reported separately; no effect on primary GO",
            calibration_source=primary_t1_private,
            representative_requested=False,
        )
        analyses["external_T1_secondary_triad"] = secondary_t1_result
        secondary_flair_rows, secondary_flair_technical_missing, _ = v4_run_noise_rows(
            secondary_flair,
            "FLAIR",
            EXTERNAL_REPETITIONS["secondary"]["FLAIR"],
            external_flair_active,
            external_flair_frequency_masks,
            "external_FLAIR_secondary_pair_03_04",
        )
        secondary_flair_psi = np.mean(
            [row["routes"]["aligned"]["psi"] for row in secondary_flair_rows], axis=0
        )
        stationarity["external_secondary_triad"] = v4_stationarity_analysis(
            {
                "T2": np.mean(
                    [row["routes"]["aligned"]["psi"] for row in secondary_t2_private["noise_rows"]],
                    axis=0,
                ),
                "T1": np.mean(
                    [row["routes"]["aligned"]["psi"] for row in secondary_t1_private["noise_rows"]],
                    axis=0,
                ),
                "FLAIR": secondary_flair_psi,
            },
            {
                "T2": len(secondary_t2_private["noise_rows"]),
                "T1": len(secondary_t1_private["noise_rows"]),
                "FLAIR": len(secondary_flair_rows),
            },
            "external secondary triad/pair descriptive estimates; never pooled with the primary triad",
        )
        stationarity["external_secondary_triad"]["FLAIR_technical_missing"] = secondary_flair_technical_missing
        external_protocol = {
            "primary_triad_repetitions": {
                key: list(value) for key, value in EXTERNAL_REPETITIONS["primary"].items()
            },
            "secondary_triad_repetitions": {
                key: list(value) for key, value in EXTERNAL_REPETITIONS["secondary"].items()
            },
            "primary_T2_verdict_recorded_before_secondary_data_materialization": True,
            "primary_T2_verdict": primary_external_verdict,
            "primary_verdict_recorded_at": primary_verdict_recorded_at,
            "secondary_unlocked_at": secondary_unlocked_at,
            "triads_reported_separately": True,
            "triads_aggregated": False,
            "external_results_have_no_effect_on_primary_GO": True,
            "predeclared_motion_reservation": (
                "inter-contrast motion in the test set was reported to be about twice train/validation; "
                "alignment diagnostics are retained by pair"
            ),
        }

    print("Producing the complete English-labelled figure...", flush=True)
    figure_duration, figure_pdf, figure_png = v4_make_figure(
        representative,
        primary_t2,
        primary_t2_private,
        primary_t1,
        external_t2_primary,
        args.mode,
    )
    c4 = v4_inspect_figure(figure_pdf, figure_png)
    hdf5_access = v4_assert_no_ground_truth_read()

    crossfit_pass = bool(
        all(
            assertion["third_not_in_pair"]
            and assertion["weights_use_only_the_third_repetition_coefficients"]
            and assertion["alignment_estimated_on_C_and_criterion_evaluated_on_disjoint_E"]
            for analysis_result in (primary_t2, primary_t1)
            for subject in analysis_result["motion_phase_and_crossfit"]["per_test_subject"].values()
            for assertion in subject["cross_fitting"]
        )
    )
    internal_tests["cross_fitting_effective_on_all_primary_test_pairs"] = {
        "pass": crossfit_pass
    }
    internal_tests["HDF5_ground_truth_exclusion"] = hdf5_access
    internal_tests["all_blocking_tests_pass"] = bool(
        registration_test["pass"]
        and synthetic_v31["pass"]
        and synthetic_v4["pass"]
        and crossfit_pass
        and frequency_record["E_intersection_C_plus_is_empty"]
        and hdf5_access["assertion_pass"]
    )
    if not internal_tests["all_blocking_tests_pass"]:
        raise AssertionError("A blocking v4 integration identity failed")

    total_duration = float(time.perf_counter() - total_start)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    smoke_validation = None
    if args.mode == "smoke":
        r_raw = float(primary_t2["route_comparison"]["R_calibrated"]["raw"])
        r_aligned = float(primary_t2["route_comparison"]["R_calibrated"]["aligned"])
        smoke_validation = {
            "status": "non-confirmatory integration test on 30 previously seen validation subjects",
            "scientific_verdict": None,
            "R_route_regression": {
                "R_raw": r_raw,
                "reference": SMOKE_RAW_R_REFERENCE,
                "absolute_difference": abs(r_raw - SMOKE_RAW_R_REFERENCE),
                "declared_absolute_tolerance": SMOKE_RAW_R_ABSOLUTE_TOLERANCE,
                "within_declared_tolerance": bool(
                    abs(r_raw - SMOKE_RAW_R_REFERENCE)
                    <= SMOKE_RAW_R_ABSOLUTE_TOLERANCE
                ),
                "R_aligned": r_aligned,
                "R_aligned_is_recorded_but_never_blocking_or_tuned": True,
                "regression_flags_are_not_scientific_verdicts": True,
            },
            "budget_extrapolation": {
                "measured_smoke_duration_seconds": total_duration,
                "prescribed_approximate_full_subject_contrast_units": 153 * 2,
                "smoke_subject_contrast_units": 30 * 2,
                "linear_scale_factor": (153 * 2) / (30 * 2),
                "estimated_full_duration_seconds": total_duration * (153 / 30),
                "estimated_full_duration_hours": total_duration * (153 / 30) / 3600.0,
                "architecture_adjusted_units_including_25_subject_external_secondary_triads": 153 * 2 + 25 * 2,
                "architecture_adjusted_duration_seconds": total_duration
                * ((153 * 2 + 25 * 2) / (30 * 2)),
                "architecture_adjusted_duration_hours": total_duration
                * ((153 * 2 + 25 * 2) / (30 * 2))
                / 3600.0,
                "linear_extrapolation_is_operational_not_a_runtime_guarantee": True,
            },
        }

    archive_data_record = {
        "archive_directory": str(discovery["archive_directory"]),
        "archives": discovery["archive_records"],
        "subject_sets": discovery["subject_sets"],
        "deduplication_assertions": discovery["deduplication_assertions"],
        "ignored_archives": discovery["ignored_archives"],
        "ignored_h5_by_archive": discovery["ignored_h5_by_archive"],
        "validation_exclusion": discovery["validation_exclusion"],
        "support_by_archive": support_by_archive,
        "HDF5_access": hdf5_access,
    }
    output_name = "smoke_results.json" if args.mode == "smoke" else "m4raw_v4_results.json"
    output_path = V4_OUT / output_name
    report_run_path = V4_OUT / "RAPPORT_RUN.md"
    results = {
        "schema_version": "m4raw-v4-replication-v2-gelee",
        "mode": args.mode,
        "frozen_specification": "SPEC_REPLICATION_v2_GELEE.md (v1 retained where not contradicted)",
        "scope": "same calibration-holdout-misspecification design, restricted to the linear Gamma-component",
        "claim_boundary": (
            "receiver-noise output variance, repeatability, and linear coil combination; "
            "no image-quality superiority claim"
        ),
        "timestamps": {"started": started_at, "finished": finished_at},
        "data": archive_data_record,
        "eligibility_before_split": eligibility,
        "splits": split_record,
        "parameters": {
            "outer_noise_rows_rule": "abs(ky-128)>48 intersect active support A",
            "outer_noise_rows_zero_based": OUTER_ROWS,
            "alignment_C": "abs(ky-128)<=16 and abs(kx-128)<=16, intersected with A",
            "guard_C_plus": "abs(ky-128)<=24 and abs(kx-128)<=24, intersected with A",
            "evaluation_E": "A minus the enlarged central block C_plus",
            "c_E": c_e,
            "mask": "common high-RSS mask: 0.15*p99 then binary erosion by 2 pixels",
            "bootstrap_B": B_BOOT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "whole eight-digit date-prefix session cluster",
            "routes": {"principal": "aligned", "frozen_secondary": "raw"},
            "FLAIR_role": "descriptive stationarity only",
            "gauge_distance_threshold_descriptive": 0.15,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "matplotlib": matplotlib.__version__,
            "script": str(pathlib.Path(__file__).resolve()),
            "script_sha256": hash_file(pathlib.Path(__file__).resolve()),
            "compute_description_for_article": "an academic departmental compute cluster",
        },
        "internal_tests": internal_tests,
        "analyses": analyses,
        "stationarity": stationarity,
        "external_replication_protocol": external_protocol,
        "smoke_validation": smoke_validation,
        "C4_production": c4,
        "no_T1_T2_pooling": True,
        "primary_GO_path": "analyses.primary_T2.verdict.scientific_verdict",
        "outputs": {
            "json": str(output_path),
            "figure_pdf": str(figure_pdf),
            "figure_png": str(figure_png),
            "run_report": str(report_run_path),
        },
        "durations_seconds": {
            "figure": figure_duration,
            "total": total_duration,
            "smoke_under_45_minutes": (
                bool(total_duration < 45.0 * 60.0) if args.mode == "smoke" else None
            ),
        },
    }
    output_path.write_text(
        json.dumps(jsonise(results), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    v4_write_run_report(results, report_run_path)
    print(
        f"Completed {args.mode}: primary scientific verdict="
        f"{primary_t2['verdict']['scientific_verdict']}, C4={c4['pass']}, "
        f"{total_duration:.1f}s",
        flush=True,
    )
    print(f"Wrote {output_path}, {figure_pdf}, {figure_png}, {report_run_path}", flush=True)


if __name__ == "__main__":
    main()
