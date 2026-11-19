"""Reduce full forward solutions to the 1284-source space used by CaliBrain.

The example forward solutions have about 5100 sources. CaliBrain works with a random
subset of 642 sources per hemisphere (1284 in total). :func:`reduce_forward_solutions`
creates it from the forward solutions in ``<data_dir>/fwd`` and writes the result to
``<data_dir>/1284src_fwd``, the folder that
:func:`calibrain.extract_leadfield.extract_leadfields` reads.
"""
import argparse
from pathlib import Path

import mne

from calibrain.utils import get_data_path, get_fwd_dir, restrict_fwd_to_sources

FULL_FWD_DIR_NAME = "fwd"  # the forward solutions with the full source space (the input)
N_SOURCES_PER_HEMISPHERE = 642  # 1284 sources in total
DEFAULT_SEED = 42  # reproduces the published 1284-source forward solutions


def reduce_forward_solutions(subjects=None, data_dir=None, seed=DEFAULT_SEED, force=False):
    """Reduce the forward solutions of several subjects to 1284 sources.

    This reads ``<data_dir>/fwd/<subject>-fwd.fif`` and writes
    ``<data_dir>/1284src_fwd/<subject>-fwd.fif``. The sources are chosen at random
    (642 per hemisphere), so the same ``seed`` always gives the same sources.

    Parameters
    ----------
    subjects : str | list of str | None
        Subjects to process. By default all subjects that have a forward solution
        (``*-fwd.fif``) in ``<data_dir>/fwd``.
    data_dir : str | Path | None
        Data directory. By default the CaliBrain data directory
        (see :func:`calibrain.utils.get_data_path`).
    seed : int, default 42
        Seed of the random choice of the sources. The default reproduces the
        1284-source forward solutions of the example datasets.
    force : bool, default False
        Overwrite reduced forward solutions that already exist. Without it, a subject
        whose reduced forward solution exists is skipped.

    Returns
    -------
    written : list of pathlib.Path
        The reduced forward solutions that were written.

    Raises
    ------
    FileNotFoundError
        If there is no forward solution to reduce.
    """
    data_dir = get_data_path(data_dir)
    full_dir = data_dir / FULL_FWD_DIR_NAME
    reduced_dir = get_fwd_dir(data_dir)

    if subjects is None:
        subjects = sorted(path.name[: -len("-fwd.fif")] for path in full_dir.glob("*-fwd.fif"))
        if not subjects:
            raise FileNotFoundError(f"No forward solutions (*-fwd.fif) found in {full_dir}.")
    elif isinstance(subjects, str):
        subjects = [subjects]

    written = []
    for subject in subjects:
        source = full_dir / f"{subject}-fwd.fif"
        if not source.is_file():
            raise FileNotFoundError(f"No forward solution for subject {subject!r} in {full_dir}.")
        target = reduced_dir / f"{subject}-fwd.fif"
        if target.exists() and not force:
            print(f"Skipping {subject}: {target} exists (use force=True to overwrite).")
            continue

        fwd = mne.read_forward_solution(str(source), verbose="error")
        reduced = restrict_fwd_to_sources(fwd, n_keep=N_SOURCES_PER_HEMISPHERE, seed=seed)
        reduced_dir.mkdir(parents=True, exist_ok=True)
        mne.write_forward_solution(str(target), reduced, overwrite=True, verbose="error")
        print(f"Reduced {subject}: {fwd['nsource']} -> {reduced['nsource']} sources")
        written.append(target)
    return written


def main(argv=None):
    """Reduce the forward solutions in the data directory to 1284 sources.

    Run with ``python -m calibrain.reduce_fwd``. By default it processes every subject
    that has a forward solution in ``<data>/fwd`` and writes ``<data>/1284src_fwd``.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", help="Data directory (default: the CaliBrain default).")
    parser.add_argument("--subjects", nargs="+", help="Subjects to process (default: all with a forward solution).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed of the random source choice (default: {DEFAULT_SEED}).")
    parser.add_argument("--force", action="store_true", help="Overwrite reduced forward solutions that exist.")
    args = parser.parse_args(argv)

    written = reduce_forward_solutions(
        subjects=args.subjects, data_dir=args.data_dir, seed=args.seed, force=args.force
    )
    print("\nWritten forward solutions:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
