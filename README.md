# auto-vertical-reframe

[English](README.md) | [繁體中文](README.zh-TW.md)

Scene-aware CLI for reframing horizontal footage into 9:16 MP4 using subject selection and a smoothed virtual camera.

## Version 0.2.2

Removes TorchCodec's unsupported `output_dtype` initialization argument. See [decoder behavior](#video-decoding-and-processing) for defaults, startup fallback, and fatal errors.

After updating a checkout, refresh installed package metadata without changing accelerator dependencies:

```bash
python3 -m pip install --no-deps -e .
```

## Features

- **Tracking:** YOLO26 segmentation and ByteTrack for people, pets, and vehicles; PySceneDetect re-evaluates subjects at cuts without guaranteed cross-cut identity.
- **Framing:** YOLO26n-pose head/body cues, subject ranking/locking, two-person framing, and smoothed pan/zoom.
- **Adaptive inference:** Optical-flow correction, pose-first composition, lightweight saliency, and on-demand DeepGaze MR; preserves every source frame.
- **Presets:** `talking_head`, `sports`, `pets`, `cars`; configurable classes and camera limits.
- **Video:** TorchCodec, optional GPU tensors, FFmpeg/NVENC encoding, audio preservation, debug overlays, and Summary telemetry.

## Quick Start

Requires **Python 3.13+** and system-installed `ffmpeg`. Python 3.12 and older are unsupported. Install inside a compatible environment:

```bash
python3 -m pip install -e .
auto_reframe.py input.mp4 output.mp4 --device auto
```

`python3 -m reframe` is an equivalent entry point. See [scenario commands](example.md); use `python3 -m reframe --help` for all options.

`requirements.txt` pins Ultralytics and tracking dependencies. Keep a single OpenCV provider (`opencv-python`): headless/contrib distributions can overwrite the shared `cv2` bindings. PyTorch/TorchVision must match each other and the accelerator build. Python version alone does not establish CUDA wheel compatibility.

### Colab installation

From a Colab code cell, with the checkout at `/content/auto-vertical-reframe`:

```python
!python3 -u /content/auto-vertical-reframe/install_colab.py
```

The installer removes legacy `mediapipe` and cached weights, replaces overlapping OpenCV providers, preserves PyTorch/TorchVision versions (including CUDA tags), and installs editable. A fresh subprocess checks imports/CUDA; the installer does not select accelerator builds. Restart the kernel if affected packages were imported.

## Video Decoding and Processing

Defaults: `--video-decoder torchcodec`, `--decode-device cuda:0`, `--decode-batch-size 4`, `--frame-processing reference`. TorchCodec needs a wheel matching PyTorch/CUDA and loadable FFmpeg shared libraries. CUDA decoding targets NVDEC; requesting CUDA is not proof of hardware placement.

| Mode | Behavior |
| :--- | :--- |
| Default / stage1 | Sequential TorchCodec decoding; reference NumPy model/render path |
| `--frame-processing tensor` / stage2 | GPU preprocessing, DeepGaze small tensor window, crop/resize; supports PyTorch `.pt` YOLO checkpoints and ByteTrack |
| `--video-decoder opencv` / baseline | Bypasses TorchCodec probing and decodes directly with OpenCV |

Both processing paths preserve source frame counts and audio synchronization. GPU and decoder commands are in [examples](example.md#decoder-and-processing-modes).

### Fallback and failure boundaries

- **Startup only:** TorchCodec import, initialization, or first-batch validation failures (`ImportError`, `OSError`, `RuntimeError`, `TypeError`, `AssertionError`) log a warning with the full native traceback and restart decoding from frame 1 using OpenCV.
- **Tensor downgrade:** Startup fallback changes requested tensor processing to `reference` (NumPy models/rendering).
- **Fatal:** VFR / `ValueError` and decode failures after first-batch validation stop processing; they do not trigger fallback or replay.
- **Timestamp validation:** TorchCodec checks sequential presentation timestamps and rejects non-CFR streams. OpenCV uses the container-reported FPS and cannot establish per-frame timestamp validation.
- **Hardware validation:** Collectors require verified NVDEC placement (`nvdec_verified`) and reject CPU or OpenCV fallback, unlike routine CLI runs.

### Reading Summary diagnostics

The `video_io` block separates intent from execution:

| Fields | Meaning |
| :--- | :--- |
| `requested_decoder`, `requested_frame_processing` | Requested pipeline |
| `decoder`, `frame_processing` | Actual pipeline after any startup fallback |
| `decoder_fallback`, `decoder_fallback_reason` | Whether startup fallback occurred and why; reason is a string or `null` |
| `cpu_fallback` | Separate native CUDA fallback diagnostic; not the OpenCV startup-fallback flag |

Summary also reports tier decisions, actual neural forward calls, cache ages, and hardware placement. Keep stdout/stderr and native tracebacks visible; a successful output file alone does not establish GPU execution.

## Pose and Subject Framing

`yolo26n-seg.pt` supplies segmentation and ByteTrack observations. `yolo26n-pose.pt` processes person ROI batches into 17 COCO keypoints. Stable tracking prioritizes the locked subject; cuts or subject competition expand pose analysis.

Head bounds come from confident nose/eye/ear keypoints, not a separate face detector; insufficient cues fall back to person bounds/masks. Turned or occluded people can use at least three reliable, geometrically sound shoulder/hip points spanning both regions. Model errors and unavailable explicitly requested model GPUs retain native tracebacks; decoder startup fallback is separate.

| Option | Default / Purpose |
| :--- | :--- |
| `--pose-model` | `yolo26n-pose.pt`; also accepts local COCO-17 person `.pt` checkpoints |
| `--pose-imgsz` | `640`; positive multiple of 32 |
| `--pose-conf` | `0.25`; pose person-box confidence |
| `--keypoint-conf` | `0.35`; individual framing-keypoint confidence |
| `--pose-batch-size` | `4`; maximum person ROIs per batch |
| `--cue-interval` | Balanced: `0.2` seconds; soft refresh, deferred when stable |
| `--pose-max-age` | Balanced: `0.35` seconds; hard cache expiry; `0` forbids cross-frame reuse |
| `--device` | `auto`; `0` requests the first CUDA GPU, `cpu` forces CPU |

## Adaptive Segmentation

With `--tracking-profile balanced`, stable detections allow candidate intervals **1 → 2 → 3 → 4 → 6 frames**, subject to the real-measurement age limit. Intervals above 3 require compatible real poses and stable mask evidence. Sparse KLT optical flow and motion prediction correct intermediate frames; cuts, lost subjects, flow failure, or intersections prompt re-detection.

| Option | Default / Purpose |
| :--- | :--- |
| `--tracking-profile` | `balanced`; alternatives: `conservative`, `fast` |
| `--seg-max-gap` | Balanced/fast: `6`; conservative: `3`; `1` disables segmentation skipping |
| `--seg-max-age` | `0.1` seconds; maximum age of real segmentation |
| `--seg-thumbnail-mode` | `lazy` builds thumbnails when needed; `eager` builds every adaptive frame |
| `--seg-thumbnail-method` | `gray-area` converts to grayscale before area resize; `bgr-area` resizes BGR first |
| `--seg-skip-policy` | `primary` lowers priority for distant low-risk secondary tracks; `all` requires all tracks settled |

Gaps are ceilings, not guaranteed skips. At the default 0.1-second age limit, 30/60 FPS permit at most 3/6 frames; 29.97/59.94 FPS permit 2/5 (floor, without rounding FPS first). Explicit scheduling flags override profile defaults regardless of flag order. `--seg-max-gap 1` does not disable pose caching; use `--pose-max-age 0` for that.

## Three-Tier Composition and DeepGaze MR

Tracking stability, pose freshness, motion, and saliency-cache validity determine each frame's tier:

| Tier | Processing |
| :--- | :--- |
| L1 `pose_only` | Reliable fresh pose and masks compose a stable primary subject without saliency inference |
| L2 `handcrafted` | Partial cues or mild changes use low-resolution spectral residual/motion differences; reliable cache may be reused |
| L3 `deepgaze` | Ambiguous focus, lost tracking, or non-human targets can require neural saliency |

A low-resolution ring buffer observes **16 consecutive source frames**, not 16 selected YOLO keyframes. DeepGaze loads and runs only when L3 needs it and the window is complete. Loading only feature weights avoids the 548 MB ImageNet VGG download; this is not a claim of faster steady-state forward passes. A stable clip may never load DeepGaze; L3 frames can reuse cache, so tier counts are not neural-call counts.

| Option | Default / Purpose |
| :--- | :--- |
| `--saliency-device` | `auto`; request `cuda` on a GPU runtime |
| `--saliency-max-side` | `384`; saliency image long-edge limit |
| `--saliency-interval` | `3`; base refresh interval when saliency is needed |
| `--saliency-amp` | FP16 autocast on by default; `--no-saliency-amp` requests FP32 |
| `--post-restore` | Off by default; opt in to FFmpeg denoising and sharpening |

## `--precrop`: Inference Region

All choices retain full height. Omit the flag for full-frame inference:

| Choice | Horizontal span | x range for 3840×2160 input |
| :--- | :--- | :--- |
| Omitted | 0%–100% | 0–3840 |
| `left` | 0%–50% | 0–1920 |
| `middle` | 25%–75% | 960–2880 |
| `right` | 50%–100% | 1920–3840 |

Detection, pose, and saliency stay inside the chosen region; the program does not search outside it if the subject leaves. Output crops the full-resolution source and can extend beyond that region. Half-width inference does not guarantee half the compute because model input resizing still applies.

## Camera and Output Controls

Use `--lock-first-subject` for the initial tracked subject, `--fixed-zoom` to disable dynamic zoom, and `--pan-time` for pan response. `--dead-zone` is a fraction of crop width/height and defaults to `0` for every preset; zoom dead zone is independent. Subject lock does not guarantee identity recovery after cuts or long occlusion. See the [fixed-zoom example](example.md#fixed-zoom-and-initial-subject) for a less reactive camera.

`--video-encoder`, `--crf`, and `--audio-bitrate` control encoding. Debug previews use `--save-debug-preview` and `--debug-path`; `--max-frames` bounds a diagnostic run. Avoid claiming speedups from fewer output frames, reduced precision, or missed subjects.

## Validation and Further Documentation

The 2026-10-13 T4 and 4K reports below describe the historical opt-in implementation, **not proof of the current default/fallback pipeline**. The user reported Colab success without new version logs. CPU passes, CUDA skips, and zero real model calls do not establish current GPU validation.

- [Agent handoff](docs/agent-handoff.md): architecture decisions, implementation status, code/test entry points, and continuation steps.
- [T4 validation specification](docs/t4-validation.md): hardware criteria, commands, and comparison gates.
- [T4 GPU report, 2026-10-13](docs/reports/t4-2026-10-13-gpu-verified.md): historical NVDEC/NVENC and inference evidence.
- [YouTube 4K report, 2026-10-13](docs/reports/youtube-validation-2026-10-13-3079cd0d9549.md): source screening, human calibration, and batch results.

Runners and notebooks: [T4 collector](scripts/colab_torchcodec.py) / [Colab](notebooks/colab-torchcodec.ipynb); [4K runner](scripts/youtube_validation.py) / [Colab](notebooks/colab-youtube-validation.ipynb). Collector dataflow `--profiles` and CLI scheduling `--tracking-profile` are separate controls.

After installing dependencies, local regressions cover geometry, pose freshness, temporal ingestion, fallback, scene resets, and FFmpeg pipes:

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest test_security test_security_reframe -v
python3 -c 'import test_torch_load; test_torch_load.test_torch_load_weights_only()'
```

## Origin and License

Built and maintained with AI. Derived from [KazKozDev/auto-vertical-reframe](https://github.com/KazKozDev/auto-vertical-reframe). [MIT License](LICENSE).
