import flexiznam as flz
import numpy as np
import pandas as pd
from twop_preprocess.pipelines import pipeline_utils

# Settings
PROJECT = "hey2_3d-vision_foodres_20220101"
MOUSE_LIST = [
    # "PZAH6.4b",
    # "PZAG3.4f",
    # "PZAH8.2i",
    # "PZAH8.2h",
    # "PZAH8.2f",
    "PZAH10.2d",
    # "PZAH10.2f",
    ]

EXCLUDE_SESSIONS = [
    'PZAH6.4b_S20220401', # not proper settings
    'PZAH6.4b_S20220411', # not proper settings
    'PZAH8.2h_S20221208', # test
    'PZAH8.2h_S20221213', # 10 depths
    'PZAH8.2h_S20221215', # test
    'PZAH8.2h_S20221216', # 10 depths
    'PZAH8.2h_S20230411x1024', # test for 10x objective
    'PZAH8.2h_S20230411x2048', # test for 10x objective
    'PZAH8.2h_S20230411', # test for 10x objective
    'PZAH8.2h_S20230410', # test for 10x objective
    'PZAH8.2i_S20221208', # test
    'PZAH8.2i_S20221209', # 10 depths
    'PZAH8.2i_S20221213', # test
    'PZAH8.2i_S20221215', # 10 depths
    'PZAH8.2f_S20221206', # test
    'PZAH8.2f_S20221209', # 10 depths
    'PZAH8.2f_S20221212', # test
    
]

PIPELINE_FILENAME = "run_suite2p_gpu.sh"
CONFLICTS = "overwrite"
TAU = 0.7

def main(
    project,
    mouse_list,
    exclude_sessions=[],
    pipeline_filename="run_suite2p_gpu.sh",
    conflicts="skip",
    tau=0.7,
):
    session_list = pipeline_utils.get_session_list(project, mouse_list, exclude_sessions)
    for session_name in session_list:
        pipeline_utils.sbatch_session(
            project=project,
            session_name=session_name,
            pipeline_filename=pipeline_filename,
            conflicts=conflicts,
            tau=tau,
        )


if __name__ == "__main__":
    main(PROJECT, MOUSE_LIST, EXCLUDE_SESSIONS, PIPELINE_FILENAME, CONFLICTS, TAU)