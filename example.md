# Command examples (Python 3.13+)

Install and check dependencies using the [README](README.md#quick-start).
Composition automatically uses pose, lightweight saliency, then on-demand DeepGaze MR.
`--saliency-model` is removed; FFmpeg denoising/sharpening require `--post-restore`.

## Decoder and processing modes

```bash
# Default: TorchCodec CUDA decoding, reference NumPy processing
python3 -m reframe input.mp4 stage1.mp4 --device 0 --saliency-device cuda

# GPU tensor processing (.pt YOLO and ByteTrack)
python3 -m reframe input.mp4 stage2.mp4 --device 0 --saliency-device cuda --frame-processing tensor

# Bypass TorchCodec probing
python3 -m reframe input.mp4 baseline.mp4 --device 0 --video-decoder opencv
```

TorchCodec startup failure logs a traceback and restarts with OpenCV from frame 1;
tensor processing downgrades to reference. VFR and midstream errors remain fatal.
See [decoder limits and telemetry](README.md#video-decoding-and-processing).

## Colab T4: central-half inference and subject lock

```bash
python3 -m reframe input.mp4 output_vertical.mp4 \
    --preset talking_head --device 0 --precrop middle \
    --lock-first-subject --saliency-device cuda --no-saliency-amp \
    --video-encoder hevc_nvenc
```

`middle` analyzes the full-height central half; output still crops the full source.
Subjects outside the inference region are not searched. See [all regions](README.md#--precrop-inference-region).
Half-width inference does not guarantee half the compute.

## Fixed zoom and initial subject

```bash
python3 -m reframe input.mp4 output_fixed.mp4 \
    --preset talking_head --lock-first-subject \
    --fixed-zoom 1.10 --dead-zone 0.08 --pan-time 0.55
```

Locks the initial tracked subject and zoom at 1.10×. The 8% crop-size pan dead zone
ignores small movements; `--pan-time` smooths recentering. Locking does not guarantee
ID recovery after cuts or long occlusion.

## Two-person podcast

```bash
python3 -m reframe podcast.mp4 output_podcast.mp4 \
    --preset talking_head --two-person-framing \
    --two-person-threshold 0.75 --switch-score-threshold 1.25 \
    --subject-hold-seconds 0.6
```

## Sports and action

```bash
python3 -m reframe skate.mp4 output_sports.mp4 \
    --preset sports --classes person bicycle motorcycle \
    --pan-time 0.1902173913043478 --pan-speed-x 0.25
```

## Debug preview and bounded validation

```bash
python3 -u -m reframe footage.mp4 output_test.mp4 \
    --preset talking_head --save-debug-preview --debug-path output_debug_hud.mp4 \
    --max-frames 90 --native-debug --log-level DEBUG --ffmpeg-log-level info
```

Compare stable subjects, crossings/occlusion, cuts, and non-person targets. Inspect
model calls, tier reasons, and crop quality, not just runtime. A stable clip may never
load DeepGaze; exercise neural saliency with an ambiguous clip longer than 16 frames.
Measure T4 speedups on matched source/settings and separate cold-start time; neither
CPU passes nor zero neural calls establish GPU validation.
