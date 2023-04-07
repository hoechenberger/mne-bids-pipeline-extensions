"""Produce power contrasts at the source level for custom frequency bands.

- Apply the inverse operator to the covariances of each condition. This yields
  a source estimate of the power.
- Calculate the difference of the log10 powers of two conditions (i.e., form a
  contrast). This yields a ratio of the log10 power: log10(cond_1 / cond_2).
- Average those contrasts across subjects, yielding a grand average power
  contrast at the source level.

This was inspired from:
https://mne.tools/stable/auto_examples/inverse/mne_cov_power.html

© Richard Höchenberger and Charbel-Raphael Segerie.


USAGE:

- Run MNE-BIDS-Pipeline completely, including the source estimation steps.
- Adjust the configuration values below
  (after "CUSTOM CONFIGURATION STARTS HERE").
- Run this script via: python source_contrasts.py

This script requires Python 3.10 or newer.


OUTPUT:

The generated output (STCs and PNG screenshots) will be placed in the
derivatives folder in a "source_contrasts" sub-directory for each subject.
Grand-average contrasts will be stored under the "sub-average" subject.
"""

import os
from pathlib import Path

import numpy as np

import mne
from mne import read_source_estimate
from mne.epochs import read_epochs
from mne.minimum_norm.inverse import apply_inverse_cov
from mne.minimum_norm import read_inverse_operator
from mne_bids import BIDSPath
from mne.source_estimate import SourceEstimate
from mne.utils import logger

from _types import FreqBand, Contrast
from _paths import get_paths


### CUSTOM CONFIGURATION STARTS HERE ###


# SPECIFY DERIVATIVES FOLDER HERE
DERIV_DIR = "~/mne_data/derivatives/mne-bids-pipeline/ds000248_no_mri/"

# SPECIFY FREESURFER SUBJECTS DIRECTORY
FS_SUBJECTS_DIR = "~/mne_data/ds000248/derivatives/freesurfer/subjects/"

# SPECIFY DATA TYPE HERE
DATA_TYPE = "meg"

# SPECIFY SUBJECTS HERE
SUBJECTS = [
    "sub-01",
]

# SPECIFY SESSION(S) HERE. PASS None IF NO SESSION NAMES ARE SPECIFIED
SESSIONS = None

# SPECIFY TASK HERE
TASK = "audiovisual"

# SPECIFY CONDITIONS TO CONTRAST HERE
CONTRASTS = [
    Contrast("Visual/Left", "Visual/Right"),
    Contrast("Visual", "Auditory"),
]

# SPECIFY FREQUENCY BANDS HERE (in Hz)
FREQ_BANDS = [
    FreqBand(name='alpha', fmin=8, fmax=12.5),
    FreqBand(name="beta", fmin=12.5, fmax=30),
    FreqBand(name="gamma", fmin=30, fmax=100),
]

# SPEFICY TIME PERIOD TO CROP EPOCHS TO. PASS None TO KEEP THE ORIGINAL EXTENT
TMIN = 0
TMAX = None

# SPECIFY NUMBER OF PARALLEL PROCESSING JOBS
N_JOBS = 1


### CUSTOM CONFIGURATION ENDS HERE ###


DERIV_DIR = Path(DERIV_DIR).expanduser()
FS_SUBJECTS_DIR = Path(FS_SUBJECTS_DIR).expanduser()
if SESSIONS is None:
    SESSIONS = [None]  # makes iteration easier

# Remove "sub-" prefix from subject names if present
SUBJECTS = [s.split("sub-")[1] if s.startswith("sub-") else s for s in SUBJECTS]

# Avoid resource oversubscription
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)


def process_one_subject(
    subject: str,
    session: str | None,
    task: str,
    datatype: str,
    deriv_root: Path,
    contrasts: list[Contrast],
    freq_bands: list[FreqBand],
    tmin: float | None,
    tmax: float | None,
    fs_subjects_dir: Path,
) -> None:
    """Compute the contrast and morph it to the fsaverage brain."""
    paths = get_paths(
        subject=subject,
        session=session,
        task=task,
        datatype=datatype,
        deriv_root=deriv_root,
    )

    # Read inverse operator
    inverse_operator = read_inverse_operator(paths["inverse_operator"])

    # Read epochs
    epochs = read_epochs(paths["epochs"])

    # Create output directory if it doesn't already exist
    assert isinstance(paths["output_dir"], Path)
    paths["output_dir"].mkdir(exist_ok=True)

    # Calculate contrasts in each frequency band
    for contrast in contrasts:
        for freq_band in freq_bands:
            logger.info(
                f"🏃 processing subject: {subject}, contrast: {contrast}, range: {freq_band.fmin}–{freq_band.fmax} Hz"
            )

            stcs = []
            for contrast_condition in ("cond_1", "cond_2"):
                contrast_condition_name = getattr(contrast, contrast_condition)

                # Filter and then crop the epochs
                epochs_condition = epochs[contrast_condition_name]
                epochs_condition_filtered = epochs_condition.filter(
                    freq_band.fmin, freq_band.fmax
                )
                epochs_condition_filtered.crop(tmin=tmin, tmax=tmax)

                cov = mne.compute_covariance(epochs_condition_filtered, rank='info')
                stc = apply_inverse_cov(
                    cov=cov,
                    info=epochs.info,
                    inverse_operator=inverse_operator,
                    nave=len(epochs_condition),
                    method="dSPM",
                )
                stc.data = np.log10(stc.data)
                stcs.append(stc)

            # Get paths again; now that we have contrast and freq_band, we can
            # retrieve the paths for the STC, too.
            paths = get_paths(
                subject=subject,
                session=session,
                task=task,
                datatype=datatype,
                deriv_root=deriv_root,
                contrast=contrast,
                freq_band=freq_band,
            )

            # Calculate the contrast
            stc_contrast = stcs[0] - stcs[1]  # Difference of logs -> ratio
            stc_contrast.save(paths["stc"], overwrite=True)

            # Plot the contrast
            brain = plot_stc(stc=stc_contrast, fs_subjects_dir=fs_subjects_dir)
            # add text to the plot
            title_string = (f"{contrast.cond_1}–{contrast.cond_2} \n"
            f"{freq_band.fmin}–{freq_band.fmax} Hz \n"
            f" {TMIN}–{TMAX} s \n")
            brain.add_text(0.7, 0.8, title_string)
            brain.save_image(paths["stc_screenshot"], mode="rgb")
            brain.close()

            # Morph the contrast to fsaverage
            morph = mne.compute_source_morph(
                stc_contrast,
                subject_from=f"sub-{subject}",
                # subject_from=None,  # ONLY FOR TESTING, DO NOT USE IN PRODUCTION
                subject_to="fsaverage",
                subjects_dir=fs_subjects_dir,
            )
            stc_contrast_morphed: SourceEstimate = morph.apply(stc_contrast)  # type: ignore
            stc_contrast_morphed.save(paths["stc_morphed"], overwrite=True)

            # Plot the morphed brain
            brain = plot_stc(stc=stc_contrast_morphed, fs_subjects_dir=fs_subjects_dir)
            # add text to the plot
            title_string = (f"{contrast.cond_1}–{contrast.cond_2} \n"
            f"{freq_band.fmin}–{freq_band.fmax} Hz \n"
            f" {TMIN}–{TMAX} s \n")
            brain.add_text(0.7, 0.8, title_string)
            brain.save_image(paths["stc_morphed_screenshot"], mode="rgb")
            brain.close()


def plot_stc(
    stc: SourceEstimate,
    fs_subjects_dir: Path,
) -> mne.viz.Brain:
    brain = stc.plot(
        subjects_dir=fs_subjects_dir,
        hemi="split",
        size=(1600, 800),
        colormap="seismic",
        clim={"kind": "percent", "pos_lims": [20, 50, 80]},
    )
    assert isinstance(brain, mne.viz.Brain)
    return brain


def grand_average(
    subjects: list[str],
    session: str | None,
    task: str,
    datatype: str,
    deriv_root: Path,
    contrasts: list[Contrast],
    freq_bands: list[FreqBand],
    fs_subjects_dir: Path,
) -> None:
    """Calculate the grand average contrasts."""

    assert subjects
    assert contrasts
    assert freq_bands

    # Calculate the grand average contrasts for all frequency bands
    for contrast in contrasts:
        for freq_band in freq_bands:
            logger.info(
                f"🏃 Grand averaging contrast: {contrast}, range: {freq_band.fmin}–{freq_band.fmax} Hz"
            )

            # Read the morphed contrasts from all subjects
            stcs_morphed = []
            for subject in subjects:
                paths = get_paths(
                    subject=subject,
                    session=session,
                    task=task,
                    datatype=datatype,
                    deriv_root=deriv_root,
                    contrast=contrast,
                    freq_band=freq_band,
                )
                stc_morphed = read_source_estimate(
                    paths["stc_morphed"], subject=subject
                )
                stcs_morphed.append(stc_morphed)

            paths = get_paths(
                subject="average",
                session=session,
                task=task,
                datatype=datatype,
                deriv_root=deriv_root,
                contrast=contrast,
                freq_band=freq_band,
            )

            # Now create a grand average contrast. This is a little hacky
            stc_grand_mean = stc_morphed.copy()
            stc_grand_mean.data = np.array([stc.data for stc in stcs_morphed]).mean(
                axis=0
            )
            stc_grand_mean.subject = "fsaverage"
            stc_grand_mean.save(paths["stc"], overwrite=True)

            # Plot the grand average contrast
            brain = plot_stc(stc=stc_grand_mean, fs_subjects_dir=fs_subjects_dir)
            # add text to the plot
            title_string = (f"{contrast.cond_1}–{contrast.cond_2} \n"
            f"{freq_band.fmin}–{freq_band.fmax} Hz \n"
            f" {TMIN}–{TMAX} s \n"
            f" N = {len(subjects)}")
            brain.add_text(0.7, 0.8, title_string)
            brain.save_image(paths["stc_screenshot"], mode="rgb")
            brain.close()


def main() -> None:
    for subject in SUBJECTS:
        for session in SESSIONS:
            process_one_subject(
                subject=subject,
                session=session,
                task=TASK,
                datatype=DATA_TYPE,
                deriv_root=DERIV_DIR,
                contrasts=CONTRASTS,
                freq_bands=FREQ_BANDS,
                tmin=TMIN,
                tmax=TMAX,
                fs_subjects_dir=FS_SUBJECTS_DIR,
            )

    for session in SESSIONS:
        grand_average(
            subjects=SUBJECTS,
            session=session,
            task=TASK,
            datatype=DATA_TYPE,
            deriv_root=DERIV_DIR,
            contrasts=CONTRASTS,
            freq_bands=FREQ_BANDS,
            fs_subjects_dir=FS_SUBJECTS_DIR,
        )


if __name__ == "__main__":
    main()
