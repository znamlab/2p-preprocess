import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("--project", "-p", help="Name of the project")
@click.option("--session", "-s", help="Flexilims name of the session")
@click.option(
    "--conflicts",
    "-c",
    default=None,
    help="How to handle conflicts when processed data already exists",
)
@click.option(
    "--run-neuropil/--no-run-neuropil", help="Whether to run ASt neuropil correction"
)
@click.option(
    "--run-split/--no-run-split", default=True, help="Whether to run split recordings"
)
@click.option(
    "--run-suite2p/--no-run-suite2p", default=True, help="Whether to suite2p extraction"
)
@click.option(
    "--run-dff/--no-run-dff", default=True, help="Whether to run dff extraction"
)
@click.option(
    "--tau", "-t", type=float, help="Decay time constant for spike extraction"
)
@click.option(
    "--keep-binary", is_flag=True, default=False, help="Whether to keep binary files"
)
@click.option(
    "--roidetect",
    type=bool,
    default=True,
    help="Whether to run ROI detection on the suite2p output",
)
def calcium(
    project,
    session,
    conflicts=None,
    run_neuropil=None,
    run_split=True,
    run_suite2p=True,
    run_dff=True,
    keep_binary=False,
    roidetect=True,
    tau=None,
):
    """Run calcium imaging preprocessing pipeline"""
    from twop_preprocess.calcium.calcium import extract_session

    ops = {
        "tau": tau,
        "ast_neuropil": run_neuropil,
        "delete_bin": not keep_binary,
        "roidetect": roidetect,
    }
    # delete None values
    ops = {k: v for k, v in ops.items() if v is not None}
    extract_session(
        project,
        session,
        conflicts=conflicts,
        run_split=run_split,
        run_suite2p=run_suite2p,
        run_dff=run_dff,
        ops=ops,
    )


@cli.command()
@click.option("--project", "-p", required=True, help="Name of the project")
@click.option("--session", "-s", required=True, help="Flexilims name of the session")
@click.option(
    "--masks-path",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the .npy file with masks to re-extract.",
)
@click.option(
    "--conflicts",
    "-c",
    default="abort",
    type=click.Choice(["abort", "skip", "append", "overwrite"]),
    help="How to handle conflicts when re-extracted data already exists.",
)
@click.option("--use-slurm/--no-use-slurm", default=True, help="Whether to use slurm")
def reextract(project, session, masks_path, conflicts, use_slurm):
    """Re-extract masks for a session."""
    from twop_preprocess.calcium.reextraction import reextract_session
    from pathlib import Path

    if use_slurm:
        slurm_folder = Path.home() / "slurm_logs" / project
        slurm_folder.mkdir(exist_ok=True, parents=True)
        scripts_name = f"reextract_{session}"
    else:
        slurm_folder = None
        scripts_name = None

    print(f"Re-extracting masks for session {session} in project {project}...")
    reextract_session(
        session,
        masks_path,
        project,
        conflicts=conflicts,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=scripts_name,
    )


@cli.command()
@click.option("--project", "-p", help="Name of the project")
@click.option("--session", "-s", help="Flexilims name of the session")
@click.option(
    "--conflicts",
    default=None,
    help="How to handle conflicts when processed data already exists",
)
@click.option("--channel", "-c", type=int, help="Channel to use for registration")
@click.option("--max-shift", type=int, help="Maximum shift for registration")
@click.option("--align-planes", type=bool, help="Whether to align planes")
@click.option("--iter", type=int, help="Number of iterations for registration")
@click.option(
    "--bidi-correction",
    type=bool,
    help="Whether to apply bidirectional correction",
)
@click.option(
    "--sequential-volumes",
    type=bool,
    help="Whether stack was imaged as a sequence of volumes rather than planes",
)
@click.option(
    "--zstack-concat",
    type=bool,
    default=False,
    help="Whether to concatenate the zstacks in datasets",
)
@click.argument("datasets", nargs=-1, required=False)
def zstack(
    project,
    session,
    conflicts=None,
    channel=None,
    max_shift=None,
    align_planes=None,
    iter=None,
    bidi_correction=None,
    sequential_volumes=None,
    datasets=None,
    zstack_concat=None,
):
    """Run zstack registration"""
    from twop_preprocess.zstack import run_zstack_registration
    from twop_preprocess.utils import load_ops

    if not datasets:
        datasets = None

    ops = {
        "ch_to_align": channel,
        "max_shift": max_shift,
        "align_planes": align_planes,
        "iter": iter,
        "bidi_correction": bidi_correction,
        "sequential_volumes": sequential_volumes,
        "zstack_concat": zstack_concat,
        "datasets": datasets,
    }
    # delete None values
    ops = {k: v for k, v in ops.items() if v is not None}
    ops = load_ops(ops, zstack=True)
    run_zstack_registration(
        project=project, session_name=session, conflicts=conflicts, ops=ops
    )
