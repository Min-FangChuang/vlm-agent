# vlm-agent

`read_test.py` reads ScanNet posed images, detects query objects, merges them into candidates with PATS matching, and sends candidate evidence to the VLM backend for final verification.

## 腳本總覽

### `read_test.py`

單筆 case 的 debug 入口。會跑目前主流程：逐 unit 建立 candidate、單 candidate VLM 驗證、`unsure` 時做 more-view completion，最後做 3D 投影與可視化。

範例：

```bash
python read_test.py --scene scene0207_00 --query chair --max-frames 10 --frame-skip 4 --max-units 3 --sam-device cuda
```

### `eval_read_test.py`

benchmark 評估入口。和 `read_test.py` 使用同一套主流程，但會遍歷 benchmark cases，並輸出 IoU 與整體統計結果。

範例：

```bash
python eval_read_test.py --case-index 0 --sam-device cuda
```

### `study_candidate_projection_completion.py`

研究用腳本，用來分析 more-view completion 流程。會選 bootstrap views、重建 3D、再投影到更多 views，並比較中間 / 最終的 3D bbox。

範例：

```bash
python study_candidate_projection_completion.py --candidate-json output/candidate_exports/scene0203_00/table/candidate_000.json --projected-json output/candidate_exports/scene0203_00/table/candidate_000_projected_views.json --sam-device cuda
```

### `project_candidate_to_other_views.py`

把單一 candidate 的 3D points 投影到場景中其他 views，輸出 projected 2D bbox，也可同時輸出 projected-point 可見性 debug 圖。

範例：

```bash
python project_candidate_to_other_views.py --candidate-json output/candidate_exports/scene0203_00/table/candidate_000.json --sam-device cuda --save-images
```

### `export_candidate_views.py`

對單一 scene/query 匯出每個 candidate 的 JSON。每個 JSON 會記錄目前 candidate 的所有 views、對應影像檔名與 2D bbox。

範例：

```bash
python export_candidate_views.py --scene scene0203_00 --query table --max-units 3
```

### `test_matcher_batches.py`

matcher 分析腳本。會以 batch 方式處理 views，印出並保存 matcher 數值、candidate 指派行為，以及 candidate 碎裂原因。

若想進一步理解 `test_matcher_batches.py` 中某一筆 match 為什麼成立、`mask_coverage` 為何會高、或 matched points 實際落在哪裡，可再搭配 `visualize_match_coverage.py` 做進階可視化檢查。

範例：

```bash
python test_matcher_batches.py --scene scene0203_00 --query table --max-units 3 --sam-device cuda
```

### `visualize_match_coverage.py`

`test_matcher_batches.py` 的進階理解工具。可針對單一 `best_view -> incoming_view` 配對，畫出 bbox 內 matches、best-view mask 支持的 matches，以及投回 incoming view 的 projected match bbox，用來分析 `mask_coverage` 為什麼成立。

圖中元素說明：

- 紅點：bbox 內的 match points
- 綠點：bbox 內且被 best-view mask 支持的 match points
- 青點：全圖 matches 中被 best-view mask 支持的點（新版 coverage 使用）
- 黃框：由青點在 incoming view 上形成的 `projected_match_bbox`
- 大綠框：incoming bbox

範例：

```bash
python visualize_match_coverage.py --scene scene0207_00 --query chair --best-view-id 00440 --best-bbox 127.05,73.55,770.15,922.53 --incoming-view-id 00480 --incoming-bbox 243.2,11.94,1081.68,968.0 --sam-device cuda
```

### `pats_test.py`

對一對影像執行 PATS matching，並輸出可視化圖與 match metadata。

範例：

```bash
python pats_test.py --image0 path/to/img0.jpg --image1 path/to/img1.jpg --draw-matches
```

### `inspect_scanrefer_case.py`

查看單一 ScanRefer benchmark case，輸出 scene/query metadata 與 GT target 資訊。

範例：

```bash
python inspect_scanrefer_case.py --case-index 0
```

### `inspect_scanrefer_target_3d.py`

讀取單一 benchmark target，抽出 GT 點雲、計算 3D bbox，並可選擇做視覺化。

範例：

```bash
python inspect_scanrefer_target_3d.py --case-index 0
```

## VLM-Grounder environment requirements

This is the WSL environment setup that has successfully run the project to the current stage.

## 1. System environment

- OS: WSL2 Ubuntu
- Python: 3.10.11
- GPU driver: use the NVIDIA driver installed on the Windows host
- WSL CUDA toolkit:
  - CUDA 11.7 for `torch 2.0.1 + cu117` and `pytorch3d` builds
  - CUDA 12.1 may coexist, but this project should build against CUDA 11.7

Install extra system packages:

```bash
sudo apt update
sudo apt install -y build-essential ninja-build git
```

## 2. Conda environment

Environment name:

```bash
vlm-grounder
```

Create and activate it:

```bash
conda create -n vlm-grounder python=3.10.11 -y
conda activate vlm-grounder
```

Install the PyTorch stack:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Confirmed working versions:

- `torch==2.0.1`
- `torch.version.cuda == 11.7`

## 3. pip / Python packages

Core packages:

```txt
numpy==1.24.4
opencv-python==4.8.0
Pillow
matplotlib==3.7.1
openai
distro
scikit-learn
defusedxml
```

OpenMMLab packages:

```txt
mmengine==0.8.2
mmcv==2.0.1
mmdet==3.3.0
openmim
```

Main project packages seen during install:

```txt
dds-cloudapi-sdk==0.2.1
gradio==4.36.1
h5py==3.8.0
httpx==0.27.0
imageio==2.33.1
imagesize==1.4.1
kornia==0.7.2
matplotlib==3.7.1
mmdet==3.3.0
mmengine==0.8.2
mmcv==2.0.1
```

Extra packages that were needed later at runtime:

```txt
typing-extensions
contourpy
cycler
pyparsing
python-dateutil
six
distro
scikit-learn
defusedxml
```

SAM:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

PyTorch3D dependencies:

```txt
fvcore
iopath
pytorch3d==0.7.8
```

Confirmed working version:

- `pytorch3d==0.7.8`

## 4. Important version constraints

### setuptools

Keep this pinned:

```txt
setuptools<82
```

Reason:

- `setuptools>=82` removes `pkg_resources`
- `mmcv`, `pytorch3d`, and `Grounding-DINO-1.5-API` builds can fail

Recommended command:

```bash
python -m pip install "setuptools<82"
```

### NumPy

Keep this pinned:

```txt
numpy==1.24.4
```

Do not use NumPy 2.x here.

Reason:

- compiled modules can break because of NumPy 2.x ABI incompatibility
- `dds-cloudapi-sdk` also requires `numpy==1.24.4`

## 5. Project-local installs

### tensor-resize / PATS setup

```bash
cd 3rdparty/pats/setup
python setup.py install
cd ../../..
```

### Grounding-DINO-1.5-API

Use no build isolation:

```bash
cd 3rdparty/Grounding-DINO-1.5-API
python -m pip install "setuptools<82"
python -m pip install --no-build-isolation -v -e .
```

### PyTorch3D

Successful build conditions:

- install CUDA 11.7 toolkit inside WSL
- build inside the Linux filesystem, not under `/mnt/c/...`
- use single-job build to reduce OOM risk

Example:

```bash
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd ~/pytorch3d
MAX_JOBS=1 python -m pip install --no-build-isolation -e .
```

## 6. Minimal saved requirements

If you want a compact list of the working environment, keep this set:

```txt
python==3.10.11
pytorch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
pytorch-cuda==11.7

numpy==1.24.4
opencv-python==4.8.0
Pillow
matplotlib==3.7.1
openai
distro
scikit-learn
defusedxml

mmengine==0.8.2
mmcv==2.0.1
mmdet==3.3.0
openmim

dds-cloudapi-sdk==0.2.1
gradio==4.36.1
h5py==3.8.0
httpx==0.27.0
imageio==2.33.1
imagesize==1.4.1
kornia==0.7.2

fvcore
iopath
pytorch3d==0.7.8

setuptools<82
```

## 7. Backend dependencies

Install Node dependencies for the VLM bridge:

```bash
cd backend
npm install
cd ..
```

Also make sure `backend/auth_data.js` is valid before calling the backend.

## 8. SAM checkpoint

Place the SAM checkpoint here:

```text
checkpoints/SAM/sam_vit_h_4b8939.pth
```

Do not commit the `.pth` file.

## 9. Data directory requirements

Make sure these dataset directories are available in the repo:

```text
scannet/posed_images/<scene>/
scannet/alignment/<scene>/
benchmark/instance_id_to_name/
benchmark/pcd_with_global_alignment/
```

## 10. Run read_test.py

From the repo root:

```bash
python read_test.py --scene scene0207_00 --query sofa --max-frames 10 --max-units 3 --sam-checkpoint checkpoints/SAM/sam_vit_h_4b8939.pth
```

Arguments:

- `--scene`: scene name under `scannet/posed_images`
- `--query`: text query passed to `Agent.reset()`
- `--max-frames`: max frames per read chunk
- `--max-units`: max chunks to process
- `--sam-checkpoint`: SAM checkpoint path
- `--sam-model-type`: SAM model type, default `vit_h`
- `--sam-device`: SAM device, e.g. `cpu` or `cuda`

`read_test.py` uses `intrinsic.txt` for projection. Do not use `depth_intrinsic.txt` here, because it can cause projection misalignment.

## 11. Output notes

Typical output fields:

- `unit`, `views`, `object_views`, `candidates`: per-chunk progress
- `[Agent] vlm_raw_result`: agent-side parsed VLM result
- `[Agent] vlm_normalized_decision`: normalized decision, one of `true`, `false`, `unsure`
- `decision`: current candidate evaluation result
- `final_selected_candidate`: final chosen candidate summary
- `bbox_3d`: projected 3D bounding box after mask completion + projection

Candidate summaries may save rendered candidate views under `output/test/`.

## 12. VLM backend note

This repo uses `backend/vlm_messages.js` to call the VLM backend.

- the Codex-mode request path may overwrite `instructions`
- to avoid losing task rules, system prompt content is merged into the first user message before sending the request
- `vlm_messages.js` stays generic and returns raw model text
- candidate-task JSON parsing and fallback now live in `agent.py`

Current agent-side fallback behavior:

- if VLM text parses as JSON, use that result
- if JSON parsing fails, convert it to:
  - `decision: "unsure"`
  - `confidence: "low"`
  - `reasoning: <raw text>`
  - empty `matched_conditions` and `missing_conditions`
  - `suggested_action: "yaw"`

## 13. Troubleshooting

- `No module named 'cv2'`
  - install `opencv-python`
- `No module named 'h5py'`
  - install `h5py`
- `segment-anything is not installed`
  - install it with `pip install git+https://github.com/facebookresearch/segment-anything.git`
- `Failed to load SAM checkpoint`
  - verify `checkpoints/SAM/sam_vit_h_4b8939.pth` exists
- `PATS config does not exist: .../pats/configs/test_scannet.yaml`
  - check that `pats/` is populated and assets are in place
- `Posed image scene directory does not exist`
  - verify `--scene` and files under `scannet/posed_images/<scene_name>/`
- backend / VLM errors
  - run `npm install` in `backend/`
  - verify `backend/auth_data.js`
