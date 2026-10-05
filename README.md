# Interactive Tree Detection

A human-in-the-loop annotation tool for detecting trees in aerial and drone
imagery. It runs [DeepForest](https://deepforest.readthedocs.io/) over a GeoTIFF,
shows you the predicted bounding boxes, lets you correct them by hand, and then
fine-tunes the model on your corrections. Each round of corrections produces a
new model version you can switch between.

A FastAPI service does the inference and training; a PyQt5 desktop application
is the annotation surface.

> **Note on the repository name.** This repo is called `monai_label` for
> historical reasons — the workflow was originally prototyped against
> [MONAI Label](https://github.com/Project-MONAI/MONAILabel). The current code
> does not use MONAI; it is built on DeepForest.

## How it works

1. **Upload** a GeoTIFF. The backend tiles it so large rasters can be panned and
   zoomed without loading the whole image into memory.
2. **Predict.** The active DeepForest model proposes bounding boxes, either over
   the whole image with a sliding window or over a single patch.
3. **Correct.** In the desktop app you add, delete and adjust boxes. Corrections
   are saved per patch.
4. **Fine-tune.** Accumulated patch annotations are turned into a training CSV
   and used to fine-tune the model. Training runs in the background and you poll
   it for progress.
5. **Switch.** The fine-tuned model is saved as a new version. You can list
   versions and switch the active one at any time.

## Layout

```
backend/
  main.py                    FastAPI app and HTTP API
  models/
    deepforest_model.py      DeepForest wrapper: prediction and fine-tuning
    model_manager.py         model versions, switching, background training
  utils/
    image_utils.py           GeoTIFF reading and tile generation
    bbox_utils.py            annotation storage and bounding-box helpers
frontend/
  pyqt_app.py                desktop application entry point
  start_pyqt_app.py          launcher that sets up Qt and the import path
  ui/
    tiff_viewer.py           pan/zoom raster canvas with box editing
    control_panel.py         prediction, annotation and training controls
    model_manager.py         model version UI
    progress_dialog.py       background task progress
  utils/
    api_client.py            HTTP client for the backend
    tiff_processor.py        client-side raster handling
```

## Requirements

- Python 3.11 or 3.13
- A CUDA-capable GPU is optional. Fine-tuning works on CPU but is slow.

## Running it

The backend and the desktop app are separate processes with separate
dependencies. Use one virtual environment for each.

**Backend:**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The API starts on `http://localhost:8000`. Interactive documentation is at
`http://localhost:8000/docs`.

**Desktop app**, in a second terminal:

```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_pyqt.txt
python start_pyqt_app.py
```

The app expects the backend at `http://localhost:8000` — see `base_url` in
`frontend/utils/api_client.py` if yours is elsewhere.

On first run the backend downloads the pretrained DeepForest release model into
`storage/models/`. That takes a moment and needs network access.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health check and active model name |
| `POST` | `/api/upload` | Upload a GeoTIFF and generate display tiles |
| `GET` | `/api/image/{image_id}/info` | Dimensions and tile metadata |
| `POST` | `/api/predict/{image_id}` | Run detection over a whole uploaded image |
| `POST` | `/api/predict_patch` | Run detection over a single patch |
| `GET` | `/api/annotations/{image_id}` | Fetch saved annotations |
| `POST` | `/api/annotations/{image_id}` | Save annotations |
| `POST` | `/api/annotations/patch/save` | Save one patch's annotations |
| `GET` | `/api/annotations/patches/count` | Count accumulated patch annotations |
| `POST` | `/api/finetune` | Fine-tune from a single annotation set |
| `POST` | `/api/finetune/patches` | Fine-tune from accumulated patches |
| `GET` | `/api/finetune/status/{task_id}` | Poll a background training task |
| `GET` | `/api/models` | List model versions |
| `POST` | `/api/models/switch` | Change the active model |

## Runtime data

Everything the app produces lives under `backend/storage/` and is git-ignored:

| Directory | Contents |
| --- | --- |
| `storage/uploads/` | uploaded GeoTIFFs |
| `storage/tiles/` | generated display tiles |
| `storage/annotations/` | saved annotations, with `patches/` for per-patch sets |
| `storage/models/` | model weights, including fine-tuned versions |
| `storage/training_data/` | generated training CSVs |

## Intended deployment

This is a **local, single-user desktop tool**. The backend has no
authentication and binds `0.0.0.0:8000`, so anything on your network can reach
it. Run it on a trusted machine and do not expose port 8000 to the internet. If
you need a shared deployment, put it behind an authenticating reverse proxy and
restrict the bind address.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go through
[SECURITY.md](SECURITY.md), not public issues.

## License

Released under the [MIT License](LICENSE). © Kesowa Infinite Ventures.

The desktop frontend depends on **PyQt5, which is GPL v3** unless you hold a
commercial licence from Riverbank Computing. Kesowa's own code here is MIT, but
if you redistribute the frontend as a combined work, GPL v3 terms apply to that
distribution. The backend has no such dependency and is MIT throughout. See
[CONTRIBUTING.md](CONTRIBUTING.md#dependency-licensing) for the detail.
