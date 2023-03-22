from pathlib import Path

from mne_bids import BIDSPath

from _types import Contrast, FreqBand, SourceContrastPaths


def get_paths(
    subject: str,
    session: str | None,
    task: str,
    datatype: str,
    deriv_root: Path,
    contrast: Contrast | None = None,
    freq_band: FreqBand | None = None,
) -> SourceContrastPaths:
    """Generate a dictionary of paths for use during processing & analysis."""

    # Always start with the clean epochs. We could pick any other existing
    # file, but epochs seem like a safe bet. The goal here is to generate a
    # BIDSPath that's un-ambiguous, to avoid any complaints by MNE-BIDS.
    bp_epochs = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        suffix="epo",
        processing="clean",
        extension=".fif",
        datatype=datatype,
        root=deriv_root,
        check=False,
    )
    bp_inv_op = bp_epochs.copy().update(processing=None, suffix="inv")
    output_dir = bp_epochs.fpath.parent / "source_contrasts"

    if contrast is not None and freq_band is not None:
        contrast_out_fname = f"sub-{subject}"
        if session is not None:
            contrast_out_fname += f"_ses-{session}"
        contrast_out_fname += f'_task-{task}_{contrast.cond_1.replace("/", "")}+{contrast.cond_2.replace("/", "")}_{freq_band.name}'
        contrast_out_path = output_dir / contrast_out_fname

        stc_path = contrast_out_path
        stc_screenshot_path = stc_path.with_suffix(".png")

        stc_morphed_path = stc_path.with_stem(stc_path.stem + "_morphed")
        stc_morphed_screenshot_path = stc_morphed_path.with_suffix(".png")
    else:
        stc_path = None
        stc_screenshot_path = None
        stc_morphed_path = None
        stc_morphed_screenshot_path = None

    return {
        "output_dir": output_dir,
        "epochs": bp_epochs,
        "inverse_operator": bp_inv_op,
        "stc": stc_path,
        "stc_screenshot": stc_screenshot_path,
        "stc_morphed": stc_morphed_path,
        "stc_morphed_screenshot": stc_morphed_screenshot_path,
    }
