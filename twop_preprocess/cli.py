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
    "--run-neuropil", is_flag=True, help="Whether to run ASt neuropil correction"
)
@click.option("--run-split", is_flag=True, help="Whether to run split recordings")
@click.option(
    "--tau", "-t", default=0.7, help="Decay time constant for spike extraction"
)
def calcium(
    project,
    session,
    conflicts=None,
    run_neuropil=True,
    run_split=True,
    tau=0.7,
):
    """Run calcium imaging preprocessing pipeline"""
    from twop_preprocess.calcium import extract_session

    ops = {
        "tau": tau,
        "ast_neuropil": run_neuropil,
    }
    extract_session(project, session, conflicts=conflicts, run_split=run_split, ops=ops)


@cli.command()
@click.option("--project", "-p", help="Name of the project")
@click.option("--session", "-s", help="Flexilims name of the session")
@click.option(
    "--conflicts",
    default=None,
    help="How to handle conflicts when processed data already exists",
)
@click.option("--channel", "-c", default=0, help="Channel to use for registration")
def zstack(project, session, conflicts=None, channel=0):
    """Run zstack registration"""
    from twop_preprocess.zstack import run_zstack_registration

    run_zstack_registration(project, session, conflicts=conflicts, ch_to_align=channel)
