from twop_preprocess.calcium import get_recording_frames, print
from twop_preprocess.utils import parse_si_metadata


import flexiznam as flz
import numpy as np


import itertools


def split_recordings(
    flz_session,
    suite2p_dataset,
    conflicts,
    base_name="suite2p_traces",
    extra_attributes=None,
):
    """
    suite2p concatenates all the recordings in a given session into a single file.
    To facilitate downstream analyses, we cut them back into chunks and add them
    to flexilims as children of the corresponding recordings.

    Args:
        flz_session (Flexilims): flexilims session
        suite2p_dataset (Dataset): dataset containing concatenated recordings
            to split
        conflicts (str): defines behavior if recordings have already been split
        base_name (str, optional): base name for the split datasets. Default
            "suite2p_traces"
        extra_attributes (dict, optional): Extra attributes to add to the split datasets
            on flexilims. Used only for identification. Default None.

    """
    # get scanimage datasets
    datasets = flz.get_datasets_recursively(
        origin_id=suite2p_dataset.origin_id,
        parent_type="recording",
        filter_parents={"recording_type": "two_photon"},
        dataset_type="scanimage",
        flexilims_session=flz_session,
        return_paths=True,
    )
    datapaths = []
    recording_ids = []
    frame_rates = []
    for recording, paths in datasets.items():
        datapaths.extend(paths)
        recording_ids.extend(itertools.repeat(recording, len(paths)))
        frame_rates.extend(
            [
                parse_si_metadata(this_path)["SI.hRoiManager.scanVolumeRate"]
                for this_path in paths
            ]
        )
    first_frames, last_frames = get_recording_frames(suite2p_dataset)
    nplanes = int(float(suite2p_dataset.extra_attributes["nplanes"]))

    datasets_out = []
    for raw_datapath, recording_id, first_frames_rec, last_frames_rec in zip(
        datapaths, recording_ids, first_frames, last_frames
    ):
        # get the path for split dataset
        split_dataset = flz.Dataset.from_origin(
            origin_id=recording_id,
            dataset_type="suite2p_traces",
            base_name=base_name,
            flexilims_session=flz_session,
            conflicts=conflicts,
        )
        # Set the extra_attributes to match that of suite2p
        # minimum number of frames across planes
        nframes = np.min(last_frames_rec - first_frames_rec)
        si_metadata = parse_si_metadata(raw_datapath)
        split_dataset.extra_attributes = suite2p_dataset.extra_attributes.copy()
        split_dataset.extra_attributes["fs"] = si_metadata[
            "SI.hRoiManager.scanVolumeRate"
        ]
        split_dataset.extra_attributes["nframes"] = nframes
        if extra_attributes is not None:
            split_dataset.extra_attributes.update(dict(extra_attributes))
        if (split_dataset.flexilims_status() == "up-to-date") and (conflicts == "skip"):
            print(f"Dataset {split_dataset.dataset_name} already exists... skipping...")
            continue

        split_dataset.path_full.mkdir(parents=True, exist_ok=True)
        np.save(split_dataset.path_full / "si_metadata.npy", si_metadata)
        # load processed data
        for iplane, start in zip(range(nplanes), first_frames_rec):
            suite2p_path = suite2p_dataset.path_full / f"plane{iplane}"
            split_path = split_dataset.path_full / f"plane{iplane}"
            try:
                split_path.mkdir(parents=True, exist_ok=True)
            except OSError:
                print(f"Error creating directory {split_path}")
            F = np.load(suite2p_path / "F.npy")
            if F.shape[1] == 0:
                print(f"No rois found when splitting recordings for plane {iplane}")
                continue
            Fneu, spks = (
                np.load(suite2p_path / "Fneu.npy"),
                np.load(suite2p_path / "spks.npy"),
            )
            end = start + nframes
            np.save(split_path / "F.npy", F[:, start:end])
            np.save(split_path / "Fneu.npy", Fneu[:, start:end])
            np.save(split_path / "spks.npy", spks[:, start:end])
            if suite2p_dataset.extra_attributes["ast_neuropil"]:
                Fast, dff_ast, spks_ast = (
                    np.load(suite2p_path / "Fast.npy"),
                    np.load(suite2p_path / "dff_ast.npy"),
                    np.load(suite2p_path / "spks_ast.npy"),
                )
                np.save(split_path / "Fast.npy", Fast[:, start:end])
                np.save(split_path / "dff_ast.npy", dff_ast[:, start:end])
                np.save(split_path / "spks_ast.npy", spks_ast[:, start:end])
            else:
                dff = np.load(suite2p_path / "dff.npy")
                np.save(split_path / "dff.npy", dff[:, start:end])
        split_dataset.update_flexilims(mode="overwrite")
        datasets_out.append(split_dataset)
    return datasets_out
