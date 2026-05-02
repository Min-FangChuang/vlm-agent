# vlm-agent

`read_test.py` reads ScanNet posed images, builds candidates with detection + PATS matching, and asks the VLM backend to verify the target object.

## Requirements

- Conda
- Python 3.11+ recommended
- Node.js for `backend/`
- ScanNet posed image data under `scannet/posed_images/<scene_name>/`
- Valid VLM auth config in `backend/auth_data.js`

Example scenes already in this repo:

- `scannet/posed_images/scene0011_00`
- `scannet/posed_images/scene0207_00`

## Conda environment

Create and activate a conda environment:

```bash
conda create -n vlm-agent python=3.11 -y
conda activate vlm-agent
```

Install Python packages used by `read_test.py`:

```bash
pip install numpy pillow opencv-python torch pyyaml ultralytics
```

These packages are needed by:

- `opencv-python`: image loading and drawing
- `torch` and `pyyaml`: PATS matcher
- `ultralytics`: `YOLOWorldDetector`
- `numpy` and `pillow`: prompt/image processing

## Install PATS

If your setup provides PATS through `3rdparty/pats/setup`, install it before running the script:

```bash
cd 3rdparty/pats/setup
python setup.py install
cd ../../..
```

The matcher also expects the local PATS assets used by `module/matcher.py`, especially:

- `pats/configs/test_scannet.yaml`
- checkpoint files referenced by that config

## Install backend dependencies

Install Node dependencies for the VLM bridge:

```bash
cd backend
npm install
cd ..
```

## Run `read_test.py`

From the repo root:

```bash
python read_test.py --scene scene0207_00 --query sofa --max-frames 10 --max-units 1
```

Arguments:

- `--scene`: scene name under `scannet/posed_images`
- `--query`: text query passed to `Agent.reset()`
- `--max-frames`: max frames per read chunk
- `--max-units`: max chunks to process

## Troubleshooting

- `No module named 'cv2'`
  - install `opencv-python`
- `PATS config does not exist: .../pats/configs/test_scannet.yaml`
  - check that `pats/` is populated and PATS assets are in place
- `Posed image scene directory does not exist`
  - verify `--scene` and the files under `scannet/posed_images/<scene_name>/`
- backend / VLM errors
  - run `npm install` in `backend/`
  - verify `backend/auth_data.js`
