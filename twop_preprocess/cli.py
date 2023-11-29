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
    "--run-neuropil", type=bool, help="Whether to run ASt neuropil correction"
)
@click.option(
    "--run-split", type=bool, default=True, help="Whether to run split recordings"
)
@click.option(
    "--run-suite2p", type=bool, default=True, help="Whether to suite2p extraction"
)
@click.option(
    "--run-dff", type=bool, default=True, help="Whether to run dff extraction"
)
@click.option(
    "--tau", "-t", type=float, help="Decay time constant for spike extraction"
)
def calcium(
    project,
    session,
    conflicts=None,
    run_neuropil=None,
    run_split=True,
    run_suite2p=True,
    run_dff=True,
    tau=None,
):
    """Run calcium imaging preprocessing pipeline"""
    from twop_preprocess.calcium import extract_session

    ops = {
        "tau": tau,
        "ast_neuropil": run_neuropil,
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
def zstack(
    project,
    session,
    conflicts=None,
    channel=None,
    max_shift=None,
    align_planes=None,
    iter=None,
    bidi_correction=None,
):
    """Run zstack registration"""
    from twop_preprocess.zstack import run_zstack_registration

    ops = {
        "ch_to_align": channel,
        "max_shift": max_shift,
        "align_planes": align_planes,
        "iter": iter,
        "bidi_correction": bidi_correction,
    }
    # delete None values
    ops = {k: v for k, v in ops.items() if v is not None}
    run_zstack_registration(project, session, conflicts=conflicts, ops=ops)
