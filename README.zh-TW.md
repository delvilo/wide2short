# auto-vertical-reframe

[English](README.md) | [繁體中文](README.zh-TW.md)

具備場景感知的 CLI，透過主角選擇與平滑虛擬相機，將橫向影片重構為 9:16 MP4。

## 版本 0.2.2

移除 TorchCodec 不支援的初始化參數 `output_dtype`。預設解碼、啟動回退與致命錯誤見[解碼行為](#解碼與影像處理)。

更新 checkout 後，刷新套件 metadata 而不變更加速器依賴：

```bash
python3 -m pip install --no-deps -e .
```

## 核心功能

- **追蹤**：YOLO26 分割與 ByteTrack 支援人物、寵物、車輛；PySceneDetect 切鏡後重新選角，不保證跨鏡頭 ID。
- **構圖**：YOLO26n-pose 頭／身體線索、主角評分／鎖定、雙人取景及平滑平移／變焦。
- **自適應推論**：光流校正、姿態優先、輕量顯著性與按需 DeepGaze MR；保留每個來源影格。
- **預設**：`talking_head`、`sports`、`pets`、`cars`；可調類別與相機限制。
- **視訊**：TorchCodec、可選 GPU tensor、FFmpeg/NVENC 編碼、保留音訊、除錯疊圖與 Summary 遙測。

## 快速開始

需要 **Python 3.13+** 與系統 `ffmpeg`；不支援 Python 3.12 或更早版本。於相容環境安裝：

```bash
python3 -m pip install -e .
auto_reframe.py input.mp4 output.mp4 --device auto
```

`python3 -m reframe` 是等效入口。情境指令見[範例](example.md)，完整選項執行 `python3 -m reframe --help`。

`requirements.txt` 固定 Ultralytics 與追蹤依賴。僅保留單一 OpenCV 提供者（`opencv-python`），避免 headless/contrib 覆寫共用 `cv2`。PyTorch/TorchVision 須彼此相容並符合加速器建置；Python 版本較新不代表有可用 CUDA wheel。

### Colab 安裝

checkout 位於 `/content/auto-vertical-reframe` 時，在 Colab code cell 執行：

```python
!python3 -u /content/auto-vertical-reframe/install_colab.py
```

安裝器移除舊 `mediapipe` 與快取權重、清除重疊 OpenCV、保留 PyTorch/TorchVision 版本（含 CUDA 標記），並以 editable 安裝。新程序檢查匯入／CUDA，但不替你選加速器建置。若已匯入受影響套件，須重啟 kernel。

## 解碼與影像處理

預設：`--video-decoder torchcodec`、`--decode-device cuda:0`、`--decode-batch-size 4`、`--frame-processing reference`。TorchCodec 需要匹配 PyTorch/CUDA 的 wheel 與可載入的 FFmpeg shared libraries。CUDA 解碼以 NVDEC 為目標，但指定 CUDA 不代表已證實硬體執行。

| 模式 | 行為 |
| :--- | :--- |
| 預設／stage1 | TorchCodec 順序解碼；NumPy reference 模型／渲染 |
| `--frame-processing tensor`／stage2 | GPU 前處理、DeepGaze 小圖 tensor 窗口、裁切／縮放；支援 PyTorch `.pt` YOLO 與 ByteTrack |
| `--video-decoder opencv`／baseline | 略過 TorchCodec 探測，直接 OpenCV 解碼 |

兩條處理路徑均保留來源影格數與音訊同步。GPU 與解碼指令見[範例](example.md#decoder-and-processing-modes)。

### 回退與錯誤邊界

- **僅啟動回退**：TorchCodec 載入、初始化或首批校驗的 `ImportError`、`OSError`、`RuntimeError`、`TypeError`、`AssertionError`，會記錄警告及完整原生 traceback，改由 OpenCV 從第 1 幀解碼。
- **Tensor 降級**：啟動回退時，要求的 tensor 處理改為 `reference`（NumPy 模型／渲染）。
- **致命錯誤**：VFR／`ValueError`、首批校驗後的解碼錯誤直接停止，不回退或重播。
- **時戳校驗**：TorchCodec 逐幀檢查顯示時戳並拒絕非 CFR；OpenCV 使用容器回報 FPS，無法建立逐幀時戳校驗。
- **硬體驗證**：收集器要求 `nvdec_verified`，拒絕 CPU 或 OpenCV 回退，嚴格度高於一般 CLI。

### 判讀 Summary

`video_io` 區分要求與實際執行：

| 欄位 | 意義 |
| :--- | :--- |
| `requested_decoder`、`requested_frame_processing` | 要求的管線 |
| `decoder`、`frame_processing` | 回退後實際管線 |
| `decoder_fallback`、`decoder_fallback_reason` | 是否啟動回退及原因；原因為字串或 `null` |
| `cpu_fallback` | 獨立的原生 CUDA 回退診斷，非 OpenCV 啟動回退旗標 |

Summary 另含層級決策、真實神經 forward 次數、快取年齡與硬體位置。保留 stdout/stderr 與原生 traceback；產出影片不等於已證實 GPU 執行。

## 姿態與主角構圖

`yolo26n-seg.pt` 提供分割與 ByteTrack 觀測；`yolo26n-pose.pt` 批次處理人物 ROI，產生 17 個 COCO 關鍵點。穩定時優先更新鎖定主角，切鏡或競爭時擴大姿態分析。

頭框由高信心鼻／眼／耳關鍵點估算，非獨立臉部偵測；不足時退回人體框／遮罩。背面或遮臉可用至少 3 個可靠、幾何合理且涵蓋肩髖的點。模型錯誤或明確要求但不可用的模型 GPU 保留原生 traceback；解碼啟動回退另論。

| 參數 | 預設／用途 |
| :--- | :--- |
| `--pose-model` | `yolo26n-pose.pt`；亦接受本機 COCO-17 人物 `.pt` 模型 |
| `--pose-imgsz` | `640`；32 的正整數倍 |
| `--pose-conf` | `0.25`；姿態人物框信心門檻 |
| `--keypoint-conf` | `0.35`；構圖關鍵點信心門檻 |
| `--pose-batch-size` | `4`；每批人物 ROI 上限 |
| `--cue-interval` | balanced：`0.2` 秒；軟更新，穩定時可延後 |
| `--pose-max-age` | balanced：`0.35` 秒；快取硬期限，`0` 禁止跨幀沿用 |
| `--device` | `auto`；`0` 指定第一張 CUDA GPU，`cpu` 強制 CPU |

## 自適應分割

`--tracking-profile balanced` 在穩定時允許候選間隔 **1 → 2 → 3 → 4 → 6 幀**，仍受真實量測年齡限制。超過 3 幀要求相容真實姿態與穩定遮罩；中間幀用稀疏 KLT 光流及運動預測校正，切鏡、主角遺失、光流失敗或交錯即提前重測。

| 參數 | 預設／用途 |
| :--- | :--- |
| `--tracking-profile` | `balanced`；另有 `conservative`、`fast` |
| `--seg-max-gap` | balanced/fast：`6`；conservative：`3`；`1` 關閉分割跳幀 |
| `--seg-max-age` | `0.1` 秒；真實分割年齡上限 |
| `--seg-thumbnail-mode` | `lazy` 按需建縮圖；`eager` 每個自適應幀皆建立 |
| `--seg-thumbnail-method` | `gray-area` 先灰階再 area 縮圖；`bgr-area` 先縮 BGR |
| `--seg-skip-policy` | `primary` 降低遠處低風險旁人優先度；`all` 要求全部穩定 |

gap 是上限而非保證跳幀。預設 0.1 秒下，30/60 FPS 最多 3/6 幀，29.97/59.94 FPS 為 2/5 幀（floor，不先四捨五入 FPS）。顯式排程參數優先於 profile，與旗標順序無關。`--seg-max-gap 1` 不停用姿態快取；需另設 `--pose-max-age 0`。

## 三級構圖與 DeepGaze MR

每幀依追蹤穩定度、姿態新鮮度、運動與顯著圖快取決定層級：

| 層級 | 處理 |
| :--- | :--- |
| L1 `pose_only` | 新鮮可靠姿態與遮罩直接構圖穩定主角，不跑顯著性 |
| L2 `handcrafted` | 部分線索或輕微變化，用低解析度頻譜殘差／運動差分；可沿用可靠快取 |
| L3 `deepgaze` | 焦點歧義、追蹤遺失或非人物目標可能需要神經顯著性 |

低解析度環形緩衝區觀測 **16 個連續來源影格**，不是 16 個 YOLO 關鍵幀。僅 L3 需要且窗口完整時載入、執行 DeepGaze。只載入特徵權重可省 548 MB ImageNet VGG 下載，不代表穩態 forward 加速。穩定片段可能不載入 DeepGaze；L3 亦可沿用快取，層級計數不等於神經呼叫次數。

| 參數 | 預設／用途 |
| :--- | :--- |
| `--saliency-device` | `auto`；GPU 環境可要求 `cuda` |
| `--saliency-max-side` | `384`；顯著圖長邊上限 |
| `--saliency-interval` | `3`；需要顯著性時的基本更新間隔 |
| `--saliency-amp` | 預設 FP16 autocast；`--no-saliency-amp` 要求 FP32 |
| `--post-restore` | 預設關閉；可選 FFmpeg 降噪與銳化 |

## `--precrop`：推論區域

所有選項皆保留全高；不指定則全畫面推論：

| 選項 | 水平範圍 | 3840×2160 輸入的 x 範圍 |
| :--- | :--- | :--- |
| 未指定 | 0%–100% | 0–3840 |
| `left` | 0%–50% | 0–1920 |
| `middle` | 25%–75% | 960–2880 |
| `right` | 50%–100% | 1920–3840 |

偵測、姿態與顯著性皆限於所選區域，主角離開時不搜尋區域外。輸出仍裁切完整解析度原圖，可延伸超出推論區域。模型仍會縮放輸入，半幅不保證計算量減半。

## 相機與輸出控制

`--lock-first-subject` 鎖定初始追蹤主角，`--fixed-zoom` 關閉動態變焦，`--pan-time` 控制平移響應。`--dead-zone` 以裁切寬／高比例表示，各 preset 預設 `0`；變焦死區獨立。鎖定不保證跨切鏡或長遮擋後恢復同一 ID。較不敏感的相機配置見[固定縮放範例](example.md#fixed-zoom-and-initial-subject)。

`--video-encoder`、`--crf`、`--audio-bitrate` 控制編碼；`--save-debug-preview` 與 `--debug-path` 輸出除錯預覽，`--max-frames` 限制診斷長度。不以減少輸出幀、降低精度或漏掉主角宣稱加速。

## 驗證與延伸文件

下列 2026-10-13 T4／4K 報表是歷史 opt-in 實作證據，**不是現行預設／回退管線的驗證證明**。使用者回報 Colab 成功但未附新版本日誌。CPU 通過、CUDA skip、模型零真實呼叫均不代表現版 GPU 通過。

- [Agent 交接](docs/agent-handoff.md)：架構決策、實作狀態、程式／測試入口與接續步驟。
- [T4 驗證規格](docs/t4-validation.md)：硬體標準、指令與比較閘門。
- [T4 GPU 報表，2026-10-13](docs/reports/t4-2026-10-13-gpu-verified.md)：歷史 NVDEC/NVENC 與推論證據。
- [YouTube 4K 報表，2026-10-13](docs/reports/youtube-validation-2026-10-13-3079cd0d9549.md)：素材篩選、人工校準與批次結果。

執行器與 notebook：[T4 收集器](scripts/colab_torchcodec.py)／[Colab](notebooks/colab-torchcodec.ipynb)；[4K 執行器](scripts/youtube_validation.py)／[Colab](notebooks/colab-youtube-validation.ipynb)。收集器資料流 `--profiles` 與 CLI 排程 `--tracking-profile` 是獨立設定。

安裝依賴後，本機回歸涵蓋幾何、姿態時效、時序觀測、回退、切鏡重置與 FFmpeg 管道：

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest test_security test_security_reframe -v
python3 -c 'import test_torch_load; test_torch_load.test_torch_load_weights_only()'
```

## 來源與授權

由 AI 建構與維護。衍生自 [KazKozDev/auto-vertical-reframe](https://github.com/KazKozDev/auto-vertical-reframe)。採 [MIT 授權](LICENSE)。
