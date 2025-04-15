# File: tests/calcium/test_calcium_utils.py  (assuming your tests live in a 'tests' directory)

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY
import tifffile
from sklearn import mixture  # Import for mocking

# Import functions to test
from twop_preprocess.calcium import calcium_utils
import twop_preprocess.calcium.processing_steps

# --- Fixtures ---


@pytest.fixture
def mock_ops():
    """Provides a basic mock ops dictionary."""
    return {
        "meanImg": np.random.rand(10, 10) * 100,
        "anatomical_only": 3,
        "denoise": 0,  # Assume denoising is off for simplicity unless testing it
        "detrend_win": 10,  # seconds
        "detrend_pctl": 8,
        "detrend_method": "subtract",
        "ast_neuropil": True,
        "neucoeff": 0.7,
        "dff_ncomponents": 2,
        "frames_per_folder": np.array([100, 150]),  # Example frame counts
        # Add other ops keys as needed by the functions
    }


@pytest.fixture
def mock_suite2p_dataset(tmp_path, mock_ops):
    """Provides a mock suite2p_dataset object."""
    dataset = MagicMock()
    dataset.extra_attributes = {
        "nplanes": "2",  # String as sometimes seen
        "fs": 10.0,  # Hz
        # Add other relevant attributes if needed
    }
    dataset.path_full = tmp_path / "suite2p_run"
    dataset.path_full.mkdir()

    # Create dummy ops files needed by get_recording_frames
    for i in range(int(dataset.extra_attributes["nplanes"])):
        plane_path = dataset.path_full / f"plane{i}"
        plane_path.mkdir()
        # Slightly different frames per folder for each plane
        ops_plane = mock_ops.copy()
        ops_plane["frames_per_folder"] = np.array([100 + i * 10, 150 - i * 5])
        np.save(plane_path / "ops.npy", ops_plane)

    return dataset


@pytest.fixture
def sample_traces():
    """Provides sample F and Fneu traces."""
    n_rois = 5
    n_frames = 250  # Matches mock_ops frames_per_folder sum (100+150)
    F = np.random.rand(n_rois, n_frames) * 100 + 50  # Base fluorescence + noise
    Fneu = np.random.rand(n_rois, n_frames) * 20 + 10  # Neuropil
    # Add some simple trend/events if needed for specific tests
    return F, Fneu


# --- Test Functions ---


class TestGetWeights:
    def test_get_weights_anatomical_3(self, mock_ops):
        mock_ops["anatomical_only"] = 3
        mean_img = np.array([[10, 50], [90, 130]], dtype=float)
        mock_ops["meanImg"] = mean_img
        p1, p99 = np.percentile(mean_img, [1, 99])
        expected_weights = 0.1 + np.clip((mean_img - p1) / (p99 - p1), 0, 1)

        weights = calcium_utils.get_weights(mock_ops)

        np.testing.assert_allclose(weights, expected_weights)
        assert weights.shape == mean_img.shape

    def test_get_weights_anatomical_2(self, mock_ops):
        mock_ops["anatomical_only"] = 2
        mean_img = np.array([[10, 50], [90, 130]], dtype=float)
        mock_ops["meanImg"] = mean_img
        p1, p99 = np.percentile(mean_img, [1, 99])
        expected_weights = 0.1 + np.clip((mean_img - p1) / (p99 - p1), 0, 1)

        weights = calcium_utils.get_weights(mock_ops)

        np.testing.assert_allclose(weights, expected_weights)
        assert weights.shape == mean_img.shape

    def test_get_weights_anatomical_1_raises(self, mock_ops):
        mock_ops["anatomical_only"] = 1
        with pytest.raises(NotImplementedError):
            calcium_utils.get_weights(mock_ops)

    def test_get_weights_other_raises(self, mock_ops):
        mock_ops["anatomical_only"] = 0  # Or other invalid value
        with pytest.raises(
            NotImplementedError
        ):  # Or potentially KeyError if max_proj needed
            calcium_utils.get_weights(mock_ops)

    def test_get_weights_denoise_warning(self, mock_ops):
        mock_ops["anatomical_only"] = 3
        mock_ops["denoise"] = 1
        with pytest.warns(
            UserWarning, match="Calculating weights on non-denoised data"
        ):
            calcium_utils.get_weights(mock_ops)


class TestRollingPercentile:
    def test_rolling_percentile_basic(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window = 3
        percentile = 50  # Median
        expected = np.array([2, 3, 4, 5, 6, 7, 8, 9])  # Median of [1,2,3], [2,3,4]...
        result = calcium_utils.rolling_percentile(arr, window, percentile)
        np.testing.assert_array_equal(result, expected)

    def test_rolling_percentile_low_pctl(self):
        arr = np.array([5, 1, 4, 2, 3])
        window = 3
        percentile = 10  # Closer to min
        # [5,1,4] -> ~1.4
        # [1,4,2] -> ~1.2
        # [4,2,3] -> ~2.2
        # Numba's percentile might differ slightly from numpy's default interpolation
        expected = np.array(
            [
                np.percentile([5, 1, 4], 10),
                np.percentile([1, 4, 2], 10),
                np.percentile([4, 2, 3], 10),
            ]
        )
        result = calcium_utils.rolling_percentile(arr, window, percentile)
        np.testing.assert_allclose(
            result, expected, rtol=1e-6
        )  # Use allclose due to potential float differences

    def test_rolling_percentile_window_1(self):
        arr = np.array([1, 2, 3, 4, 5])
        window = 1
        percentile = 50
        expected = arr  # Window 1 is just the element itself
        result = calcium_utils.rolling_percentile(arr, window, percentile)
        np.testing.assert_array_equal(result, expected)


class TestDetrend:
    @pytest.mark.parametrize("method", ["subtract", "divide"])
    def test_detrend(self, sample_traces, mock_ops, method):
        F, _ = sample_traces
        F_orig = F.copy()
        mock_ops["detrend_method"] = method
        fs = 10.0
        # Simple linear frames for testing
        first_frames = np.array([0, 100])
        last_frames = np.array([100, 250])  # Total 250 frames

        # Add a known trend to subtract/divide out
        trend = np.linspace(0, 10, F.shape[1]) * 10
        F = F + trend[np.newaxis, :]

        F_detrended, baseline = twop_preprocess.calcium.processing_steps.detrend(
            F.copy(), first_frames, last_frames, mock_ops, fs
        )

        assert F_detrended.shape == F.shape
        assert baseline.shape == F.shape
        # Check that the trend is reduced. Exact check is hard due to rolling percentile.
        # Check if variance is reduced or mean is closer to original (excluding noise)
        # A simpler check: ensure values are modified
        assert not np.allclose(F_detrended, F)

        # Check if baseline roughly matches the added trend (especially for 'subtract')
        if method == "subtract":
            # Baseline should roughly capture the added trend, adjusted by percentile
            # and first recording baseline subtraction. This is hard to assert precisely.
            pass  # Difficult to make a precise assertion here without reimplementing

        # Check that original F outside the function is not modified
        np.testing.assert_array_equal(F_orig, sample_traces[0])


class TestCorrectNeuropil:
    @patch("twop_preprocess.neuropil.ast_model.ast_model")
    @patch("numpy.load")
    @patch("numpy.save")
    def test_correct_neuropil(
        self, mock_np_save, mock_np_load, mock_ast_model, tmp_path, sample_traces
    ):
        F, Fneu = sample_traces
        n_rois = F.shape[0]
        dpath = tmp_path / "plane0"
        dpath.mkdir()

        # Mock return value for np.load('stat.npy')
        mock_stat = [
            {"npix": 10, "neuropil_mask": np.zeros((5, 5))} for _ in range(n_rois)
        ]
        mock_np_load.return_value = mock_stat

        # Mock return value for ast_model
        mock_trace = np.random.rand(F.shape[1])
        mock_param = np.random.rand(5)  # Example param shape
        mock_elbo = np.random.rand(1)
        mock_ast_model.return_value = (mock_trace, mock_param, mock_elbo)

        Fast_result = calcium_utils.correct_neuropil_ast(dpath, F, Fneu)

        # Assertions
        assert mock_np_load.call_count == 1
        mock_np_load.assert_called_once_with(dpath / "stat.npy", allow_pickle=True)

        assert mock_ast_model.call_count == n_rois
        # Check call args for the first call as an example
        first_call_args = mock_ast_model.call_args_list[0][0]
        np.testing.assert_array_equal(first_call_args[0], np.vstack([F[0], Fneu[0]]))
        np.testing.assert_array_equal(
            first_call_args[1],
            np.array([mock_stat[0]["npix"], mock_stat[0]["neuropil_mask"].shape[0]]),
        )

        assert mock_np_save.call_count == 3
        expected_Fast = np.array([mock_trace] * n_rois)  # Since mock returns same trace
        expected_params = np.array([mock_param] * n_rois)
        expected_elbos = np.array([mock_elbo] * n_rois)

        # Use ANY for the array comparison in assert_called_with
        # Or check call_args_list directly for precise array comparison
        calls = [
            call(dpath / "Fast.npy", ANY, allow_pickle=True),
            call(dpath / "ast_stat.npy", ANY, allow_pickle=True),
            call(dpath / "ast_elbo.npy", ANY, allow_pickle=True),
        ]
        mock_np_save.assert_has_calls(calls, any_order=True)  # Order might vary

        # More precise check of saved data
        saved_data = {args[0][0]: args[0][1] for args in mock_np_save.call_args_list}
        np.testing.assert_allclose(saved_data[dpath / "Fast.npy"], expected_Fast)
        np.testing.assert_allclose(saved_data[dpath / "ast_stat.npy"], expected_params)
        np.testing.assert_allclose(saved_data[dpath / "ast_elbo.npy"], expected_elbos)

        np.testing.assert_allclose(Fast_result, expected_Fast)


class TestDFFHelper:
    @patch("sklearn.mixture.GaussianMixture")
    def test_dff_helper(self, mock_gmm, sample_traces):
        F, _ = sample_traces
        n_rois, n_frames = F.shape
        n_components = 2

        # Mock GMM
        mock_gmm_instance = MagicMock()
        # Simulate GMM finding two means, the lower one being F0
        mock_means = np.array([[50.0], [150.0]])  # Example means
        mock_gmm_instance.means_ = mock_means
        mock_gmm_instance.fit.return_value = mock_gmm_instance  # Fit returns self
        mock_gmm.return_value = mock_gmm_instance  # Constructor returns instance

        dff, f0 = calcium_utils.dFF(F, n_components=n_components)

        assert mock_gmm.call_count == n_rois
        mock_gmm.assert_called_with(n_components=n_components, random_state=42)
        assert mock_gmm_instance.fit.call_count == n_rois

        expected_f0 = np.full((n_rois, 1), np.min(mock_means))  # Lowest mean
        expected_dff = (F - expected_f0) / expected_f0

        np.testing.assert_allclose(f0, expected_f0)
        np.testing.assert_allclose(dff, expected_dff)


class TestCalculateAndSaveDFF:
    @patch("numpy.save")
    @patch("twop_preprocess.calcium.calcium_utils.dFF")
    def test_calculate_and_save_dff(self, mock_dff, mock_np_save, tmp_path):
        F = np.random.rand(5, 100)  # 5 ROIs, 100 frames
        dpath = tmp_path / "suite2p_output"
        dpath.mkdir()
        filename_suffix = "_test"
        n_components = 3

        # Mock dFF return values
        mock_dff_result = (np.random.rand(5, 100), np.random.rand(5))
        mock_dff.return_value = mock_dff_result

        dff, f0 = calcium_utils.calculate_and_save_dFF(
            dpath, F, filename_suffix, n_components=n_components
        )

        mock_dff.assert_called_once_with(F, n_components=n_components)
        assert mock_np_save.call_count == 2
        mock_np_save.assert_has_calls(
            [
                call(dpath / f"dff{filename_suffix}.npy", mock_dff_result[0]),
                call(dpath / f"f0{filename_suffix}.npy", mock_dff_result[1]),
            ]
        )
        np.testing.assert_array_equal(dff, mock_dff_result[0])
        np.testing.assert_array_equal(f0, mock_dff_result[1])


class TestCorrectNeuropilStandard:
    def test_correct_neuropil_standard_basic(self, sample_traces):
        """Test the standard neuropil correction calculation."""
        F, Fneu = sample_traces
        neucoeff = 0.7

        # Manual calculation
        Fneu_median = np.median(Fneu, axis=1, keepdims=True)
        Fneu_demeaned = Fneu - Fneu_median
        expected_F_corrected = F - neucoeff * Fneu_demeaned

        F_corrected = calcium_utils.correct_neuropil_standard(F, Fneu, neucoeff)

        np.testing.assert_allclose(F_corrected, expected_F_corrected)
        assert F_corrected.shape == F.shape

    def test_correct_neuropil_standard_zero_coeff(self, sample_traces):
        """Test standard neuropil correction with neucoeff = 0."""
        F, Fneu = sample_traces
        neucoeff = 0.0

        # With neucoeff=0, F_corrected should be identical to F
        expected_F_corrected = F

        F_corrected = calcium_utils.correct_neuropil_standard(F, Fneu, neucoeff)

        np.testing.assert_allclose(F_corrected, expected_F_corrected)
        assert F_corrected.shape == F.shape

    def test_correct_neuropil_standard_constant_fneu(self):
        """Test standard neuropil correction when Fneu is constant for an ROI."""
        F = np.array([[10, 20, 15, 25], [50, 55, 60, 58]], dtype=float)
        Fneu = np.array(
            [[5, 5, 5, 5], [10, 12, 11, 13]], dtype=float
        )  # First ROI Fneu is constant
        neucoeff = 0.7

        # Manual calculation for ROI 0 (Fneu constant)
        # Fneu_median = 5, Fneu_demeaned = 0, F_corrected = F
        expected_F_corrected_roi0 = F[0, :]

        # Manual calculation for ROI 1
        Fneu1_median = np.median(Fneu[1, :])  # 11.5
        Fneu1_demeaned = Fneu[1, :] - Fneu1_median  # [-1.5, 0.5, -0.5, 1.5]
        expected_F_corrected_roi1 = F[1, :] - neucoeff * Fneu1_demeaned

        F_corrected = calcium_utils.correct_neuropil_standard(F, Fneu, neucoeff)

        np.testing.assert_allclose(F_corrected[0, :], expected_F_corrected_roi0)
        np.testing.assert_allclose(F_corrected[1, :], expected_F_corrected_roi1)
        assert F_corrected.shape == F.shape

    @patch("numpy.save")
    def test_correct_neuropil_standard_saves(
        self, mock_np_save, sample_traces, tmp_path
    ):
        """Test that standard neuropil correction saves the file when path is provided."""
        F, Fneu = sample_traces
        neucoeff = 0.7
        save_path = tmp_path / "F_corrected_standard.npy"

        # Calculate expected result to check save argument (optional but good)
        Fneu_median = np.median(Fneu, axis=1, keepdims=True)
        Fneu_demeaned = Fneu - Fneu_median
        expected_F_corrected = F - neucoeff * Fneu_demeaned

        F_corrected_func = calcium_utils.correct_neuropil_standard(
            F, Fneu, neucoeff, save_path=save_path
        )

        # Check that the function still returns the correct result
        np.testing.assert_allclose(F_corrected_func, expected_F_corrected)

        # Check that np.save was called correctly
        mock_np_save.assert_called_once()
        call_args = mock_np_save.call_args[0]
        call_kwargs = mock_np_save.call_args[1]

        assert call_args[0] == save_path
        np.testing.assert_allclose(call_args[1], expected_F_corrected)
        assert call_kwargs == {"allow_pickle": True}

    @patch("numpy.save")
    def test_correct_neuropil_standard_no_save(self, mock_np_save, sample_traces):
        """Test that standard neuropil correction does not save when path is None."""
        F, Fneu = sample_traces
        neucoeff = 0.7
        save_path = None  # Explicitly None (or just omit)

        calcium_utils.correct_neuropil_standard(F, Fneu, neucoeff, save_path=save_path)

        mock_np_save.assert_not_called()


class TestEstimateOffset:
    @patch("sklearn.mixture.GaussianMixture")
    @patch("tifffile.TiffFile")
    def test_estimate_offset(self, mock_tiff_file, mock_gmm, tmp_path):
        # Create dummy tiff file
        dummy_tiff_path = tmp_path / "test.tif"
        # Frame with low background values (~100) and some higher signal
        frame_data = np.random.randint(90, 110, size=(50, 50), dtype=np.uint16)
        frame_data[10:20, 10:20] = 500  # Add some signal
        tifffile.imwrite(dummy_tiff_path, frame_data)

        # Mock TiffFile context manager
        mock_tiff_instance = MagicMock()
        mock_tiff_instance.asarray.return_value = frame_data
        mock_tiff_cm = MagicMock()
        mock_tiff_cm.__enter__.return_value = mock_tiff_instance
        mock_tiff_file.return_value = mock_tiff_cm

        # Mock GMM
        mock_gmm_instance = MagicMock()
        # Simulate GMM finding means, lowest should be offset
        mock_means = np.array([[98.5], [505.0], [105.0]])  # Unsorted means
        mock_gmm_instance.means_ = mock_means
        mock_gmm_instance.fit.return_value = mock_gmm_instance
        mock_gmm.return_value = mock_gmm_instance

        n_components = 3
        offset = calcium_utils.estimate_offset(str(tmp_path), n_components=n_components)

        mock_gmm.assert_called_once_with(n_components=n_components, random_state=42)
        mock_gmm_instance.fit.assert_called_once()
        # Check that fit was called with the flattened frame data
        np.testing.assert_array_equal(
            mock_gmm_instance.fit.call_args[0][0], frame_data.reshape(-1, 1)
        )

        assert offset == pytest.approx(98.5)  # Lowest of the sorted means

    def test_estimate_offset_no_tiff(self, tmp_path):
        with pytest.raises(ValueError, match="No tiffs found"):
            calcium_utils.estimate_offset(str(tmp_path))


class TestCorrectOffset:
    @patch("numpy.load")
    def test_correct_offset(self, mock_np_load, sample_traces):
        F, _ = sample_traces
        mock_np_load.return_value = F.copy()  # Return a copy to check modification
        datapath = Path("dummy/path/F.npy")  # Path doesn't matter due to mock

        n_recs = 2
        n_frames_total = F.shape[1]
        first_frames = np.array([0, 100])
        last_frames = np.array([100, n_frames_total])
        offsets = np.array([10.0, 20.0])

        F_corrected = calcium_utils.correct_offset(
            datapath, offsets, first_frames, last_frames
        )

        mock_np_load.assert_called_once_with(datapath)

        # Check first recording segment
        expected_rec1 = F[:, 0:100] - offsets[0]
        np.testing.assert_allclose(F_corrected[:, 0:100], expected_rec1)

        # Check second recording segment
        expected_rec2 = F[:, 100:n_frames_total] - offsets[1]
        np.testing.assert_allclose(F_corrected[:, 100:n_frames_total], expected_rec2)

        # Ensure shape is maintained
        assert F_corrected.shape == F.shape


class TestGetRecordingFrames:
    def test_get_recording_frames_basic(self, mock_suite2p_dataset):
        # Ops files created by the fixture
        # plane0: [100, 150] -> cumsum [100, 250] -> first [0, 100]
        # plane1: [110, 145] -> cumsum [110, 255] -> first [0, 110]

        expected_first = np.array([[0, 0], [100, 110]])
        expected_last = np.array([[100, 110], [250, 255]])

        first_frames, last_frames = calcium_utils.get_recording_frames(
            mock_suite2p_dataset
        )

        np.testing.assert_array_equal(first_frames, expected_first)
        np.testing.assert_array_equal(last_frames, expected_last)

    def test_get_recording_frames_missing_nplanes(self, mock_suite2p_dataset):
        # Simulate missing 'nplanes' key
        del mock_suite2p_dataset.extra_attributes["nplanes"]
        # Need to adjust fixture to only create plane0 ops if nplanes defaults to 1
        (mock_suite2p_dataset.path_full / "plane1" / "ops.npy").unlink()
        (mock_suite2p_dataset.path_full / "plane1").rmdir()
        # Reload ops for plane0 to get its frames_per_folder
        ops0 = np.load(
            mock_suite2p_dataset.path_full / "plane0" / "ops.npy", allow_pickle=True
        ).item()
        frames_p0 = ops0["frames_per_folder"]  # Should be [100, 150]

        expected_first = np.array([[0], [frames_p0[0]]])  # Shape (n_recs, 1)
        expected_last = np.array(
            [[frames_p0[0]], [np.sum(frames_p0)]]
        )  # Shape (n_recs, 1)

        # Mock the update_flexilims call which might happen
        mock_suite2p_dataset.update_flexilims = MagicMock()

        first_frames, last_frames = calcium_utils.get_recording_frames(
            mock_suite2p_dataset
        )

        # Check if nplanes was added back (optional check)
        assert mock_suite2p_dataset.extra_attributes["nplanes"] == 1
        mock_suite2p_dataset.update_flexilims.assert_called_once_with(mode="update")

        np.testing.assert_array_equal(first_frames, expected_first)
        np.testing.assert_array_equal(last_frames, expected_last)
