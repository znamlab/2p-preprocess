import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy
import flexiznam as flz
import subprocess
import shlex
from functools import partial
print = partial(print, flush=True)


def sbatch_session(
    project, session_name, pipeline_filename, conflicts, tau=0.7
):
    """Start sbatch script to run analysis_pipeline on a single session.

    Args:

    """

    script_path = str(
        Path(__file__).parent.parent.parent / pipeline_filename
    )

    log_fname = f"{session_name}_%j.log"

    log_path = str(Path(__file__).parent.parent.parent / "logs" / f"{log_fname}")

    args = f"--export=PROJECT={project},SESSION={session_name},CONFLICTS={conflicts},TAU={tau}"

    args = args + f" --output={log_path}"

    command = f"sbatch {args} {script_path}"
    print(command)
    subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    

def get_session_list(project, mouse_list, exclude_sessions=[]):
    session_list = []
    flexilims_session = flz.get_flexilims_session(project)
    for mouse in mouse_list:
        sessions_mouse = flz.get_children(parent_name=mouse, children_datatype="session", flexilims_session=flexilims_session).name.values.tolist()
        session_list.append(sessions_mouse)
    
    # exclude any sessions from exclude_sessions
    session_list = [session for i in session_list for session in i]
    session_list = [session for session in session_list if session not in exclude_sessions]
    
    return session_list