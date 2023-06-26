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
@click.option("--run-neuropil", is_flag=True, help="Whether to run neuropil correction")
@click.option("--run-split", is_flag=True, help="Whether to run split recordings")
@click.option(
    "--tau", "-t", default=0.7, help="Decay time constant for spike extraction"
)
@click.option("--nplanes", default=1, help="Number of planes in the recording")
@click.option(
    "--dff_ncomponents",
    default=2,
    help="Number of GMM components to use for dF/F calculation",
)
def calcium(
    project,
    session,
    conflicts=None,
    run_neuropil=False,
    run_split=False,
    tau=0.7,
    nplanes=1,
    dff_ncomponents=2,
):
    """Run calcium imaging preprocessing pipeline"""
    from twop_preprocess.calcium import extract_session

    extract_session(
        project,
        session,
        conflicts=conflicts,
        run_neuropil=run_neuropil,
        run_split=run_split,
        tau=tau,
        nplanes=nplanes,
        dff_ncomponents=dff_ncomponents,
    )


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
