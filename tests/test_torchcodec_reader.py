"""Timestamp, memory-bound, and accelerator guarantees of optional decoding."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reframe.torchcodec_reader import TorchCodecReader


class FallbackStatus:
    def __init__(self, fallback=False, known=True):
        self.status_known = known
        self.fallback = fallback

    def __bool__(self):
        return self.fallback

    def __str__(self):
        return "mock CPU fallback" if self.fallback else "mock hardware decoder"


class SyntheticDecoder:
    def __init__(self, count=11, fps=30., start=2.5, pts_fps=None, skip_slot_at=None):
        self.metadata = SimpleNamespace(
            average_fps=fps,
            average_fps_from_header=fps,
            width=32, height=24, num_frames=count,
            begin_stream_seconds=start,
        )
        self.pts_fps = pts_fps if pts_fps is not None else fps
        self.skip_slot_at = skip_slot_at
        self.cpu_fallback = FallbackStatus()
        self.calls = []
        self.pts_offsets = {}

    def get_frames_in_range(self, start, stop):
        self.calls.append((start, stop))
        data = torch.arange(start, stop, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 24, 32)
        pts = []
        for i in range(start, stop):
            pts_sec = self.metadata.begin_stream_seconds + i / self.pts_fps
            if self.skip_slot_at is not None and i >= self.skip_slot_at:
                pts_sec += 1.0 / self.pts_fps  # Missing PTS slot shifts by one frame
            pts.append(pts_sec + self.pts_offsets.get(i, 0))
        return SimpleNamespace(data=data, pts_seconds=torch.tensor(pts, dtype=torch.float64))


class TorchCodecReaderTests(unittest.TestCase):
    def open_reader(self, decoder=None, **kwargs):
        decoder = decoder or SyntheticDecoder()
        self.enterContext(patch("reframe.torchcodec_reader._open_decoder", return_value=decoder))
        self.enterContext(patch("reframe.torchcodec_reader.version", return_value="0.17.0-test"))
        reader = TorchCodecReader("test.mp4", **kwargs)
        self.addCleanup(reader.close)
        return reader, decoder

    def test_all_frames_in_order_with_partial_last_batch(self):
        reader, decoder = self.open_reader(batch_size=4)
        frames = list(reader)
        self.assertEqual([index for index, _ in frames], list(range(1, 12)))
        self.assertEqual([int(frame.rgb[0, 0, 0]) for _, frame in frames], list(range(11)))
        self.assertEqual(decoder.calls, [(0, 4), (4, 8), (8, 11)])
        self.assertTrue(all(frame.shape == (24, 32, 3) for _, frame in frames))
        stats = reader.telemetry()
        self.assertEqual(stats["frames_decoded"], 11)
        self.assertEqual(stats["frames_yielded"], 11)
        self.assertEqual(stats["max_batch_frames"], 4)
        self.assertEqual(stats["actual_device"], "cpu")
        self.assertIs(stats["cpu_fallback"], False)
        json.dumps(stats)

    def test_early_close_decodes_only_one_small_batch(self):
        reader, decoder = self.open_reader(SyntheticDecoder(count=100000), batch_size=8)
        self.assertEqual(next(reader)[0], 1)
        reader.close()
        reader.close()
        self.assertEqual(decoder.calls, [(0, 8)])
        self.assertEqual(reader.telemetry()["frames_decoded"], 8)
        self.assertEqual(reader.telemetry()["frames_yielded"], 1)
        self.assertEqual(list(reader), [])

    def test_variable_fps_and_discontinuous_pts_rejected(self):
        for offset in (-.01, .01, float("nan")):
            with self.subTest(offset=offset):
                decoder = SyntheticDecoder()
                decoder.pts_offsets[2] = offset
                reader, _ = self.open_reader(decoder)
                with self.assertRaisesRegex(ValueError, "Variable-frame-rate.*frame 3"):
                    next(reader)

    def test_container_timestamp_rounding_is_allowed(self):
        decoder = SyntheticDecoder(fps=30000 / 1001)
        decoder.pts_offsets = {i: .0004 * (-1) ** i for i in range(11)}
        reader, _ = self.open_reader(decoder)
        self.assertEqual(len(list(reader)), 11)

    def test_header_rate_overrides_content_rate_for_pts_grid(self):
        # Container content-scanned rate (59.9937) differs from header rate (60).
        # PTS are on the 60 Hz grid; reader must use header rate and accept all frames.
        decoder = SyntheticDecoder(
            count=9516, fps=59.99369549227698, pts_fps=60.0, start=0.0
        )
        decoder.metadata.average_fps_from_header = 60.0
        decoder.pts_offsets = {}
        reader, _ = self.open_reader(decoder)
        frames = list(reader)
        self.assertEqual(len(frames), 9516)
        self.assertEqual(reader.fps, 60.0)
        self.assertEqual(reader.telemetry()["fps_source"], "header")
        self.assertEqual(reader.telemetry()["source_fps"], 60.0)
        self.assertEqual(reader.telemetry()["metadata_fps"], 59.99369549227698)
    def test_missing_pts_slot_mid_stream_accepted(self):
        # Container drops one PTS slot at frame 5; all subsequent PTS shift by
        # one frame duration. Reader must accept all frames and carry offset.
        decoder = SyntheticDecoder(count=20, fps=30.0, skip_slot_at=5)
        reader, _ = self.open_reader(decoder, batch_size=4)
        frames = list(reader)
        self.assertEqual(len(frames), 20)
        tel = reader.telemetry()
        self.assertEqual(tel["gap_count"], 1)
        self.assertEqual(tel["pts_offset_frames"], 1)

    def test_missing_pts_slot_near_end_accepted(self):
        # Missing slot near end of stream.
        decoder = SyntheticDecoder(count=20, fps=30.0, skip_slot_at=18)
        reader, _ = self.open_reader(decoder, batch_size=4)
        frames = list(reader)
        self.assertEqual(len(frames), 20)
        tel = reader.telemetry()
        self.assertEqual(tel["gap_count"], 1)
        self.assertEqual(tel["pts_offset_frames"], 1)

    def test_too_many_consecutive_gaps_rejected(self):
        # Eleven missing PTS slots exceed the total gap cap of 10.
        decoder = SyntheticDecoder(count=20, fps=30.0)
        decoder.pts_offsets = {i: 1.0 / 30.0 for i in range(11)}
        reader, _ = self.open_reader(decoder, batch_size=4)
        with self.assertRaisesRegex(ValueError, "Variable-frame-rate"):
            list(reader)

    def test_single_large_gap_rejected(self):
        # Single gap of 5 slots exceeds per-gap cap of 4.
        decoder = SyntheticDecoder(count=20, fps=30.0)
        decoder.pts_offsets = {5: 5.0 / 30.0}
        reader, _ = self.open_reader(decoder, batch_size=4)
        with self.assertRaisesRegex(ValueError, "Variable-frame-rate"):
            list(reader)

    def test_cuda_fallback_or_unknown_status_rejected_before_cpu_tensor_is_used(self):
        for status in (FallbackStatus(fallback=True), FallbackStatus(known=False)):
            with self.subTest(status=status):
                decoder = SyntheticDecoder()
                decoder.cpu_fallback = status
                reader, _ = self.open_reader(decoder, device="cuda:0")
                with self.assertRaisesRegex(RuntimeError, "hardware decoding was not verified"):
                    next(reader)
                self.assertEqual(reader.telemetry()["frames_yielded"], 0)

    def test_cuda_request_cannot_silently_return_cpu_pixels(self):
        reader, _ = self.open_reader(device="cuda:0")
        with self.assertRaisesRegex(RuntimeError, "requested cuda:0 but returned cpu"):
            next(reader)

    def test_native_decoder_error_is_preserved(self):
        reader, decoder = self.open_reader()
        failure = RuntimeError("FFmpeg original decoder error")
        decoder.get_frames_in_range = lambda **kwargs: (_ for _ in ()).throw(failure)
        with self.assertRaises(RuntimeError) as caught:
            next(reader)
        self.assertIs(caught.exception, failure)

    def test_invalid_batch_size_fails_before_decoder_loading(self):
        with patch("reframe.torchcodec_reader._open_decoder") as open_decoder:
            for value in (0, 9, -1, 1.5, True):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "1 to 8"):
                    TorchCodecReader("test.mp4", batch_size=value)
            open_decoder.assert_not_called()

    def test_invalid_metadata_rejected(self):
        for field, value in (("average_fps", 0), ("average_fps", float("nan")),
                             ("num_frames", 0), ("width", 0)):
            with self.subTest(field=field, value=value):
                decoder = SyntheticDecoder()
                setattr(decoder.metadata, field, value)
                if field == "average_fps":
                    # The header rate is the reference grid; both sources must be invalid.
                    decoder.metadata.average_fps_from_header = value
                with self.assertRaisesRegex(ValueError, "invalid video"):
                    self.open_reader(decoder)

    def test_import_does_not_require_torchcodec(self):
        # `_open_decoder` is the only dependency entry point.  Merely using
        # OpenCV/default CLI paths must not initialize native TorchCodec libs.
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import reframe.torchcodec_reader; "
             "assert 'torchcodec' not in sys.modules; assert 'torch' not in sys.modules"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(importlib.util.find_spec("torchcodec"), "Optional TorchCodec is not installed")
class RealTorchCodecReaderTests(unittest.TestCase):
    def test_real_mp4_preserves_fps_frame_count_and_rgb_order(self):
        import cv2

        with TemporaryDirectory() as directory:
            path = Path(directory) / "colored.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(11):
                    bgr = np.empty((48, 64, 3), np.uint8)
                    bgr[:] = (20 + index * 10, 70, 180)
                    writer.write(bgr)
            finally:
                writer.release()
            with TorchCodecReader(path, batch_size=4) as reader:
                self.assertAlmostEqual(reader.fps, 30)
                self.assertEqual(reader.total_frames, 11)
                colors = []
                indices = []
                for index, frame in reader:
                    indices.append(index)
                    colors.append(frame.rgb[:, 24, 32].tolist())
                self.assertEqual(indices, list(range(1, 12)))
                self.assertTrue(all(abs(color[0] - 180) < 8 for color in colors))
                self.assertTrue(all(abs(color[1] - 70) < 8 for color in colors))
                self.assertTrue(all(abs(color[2] - (20 + i * 10)) < 8 for i, color in enumerate(colors)))
                self.assertEqual(reader.telemetry()["frames_yielded"], 11)


if __name__ == "__main__":
    unittest.main()
