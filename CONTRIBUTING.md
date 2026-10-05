# Contributing

Thanks for your interest. This is a research and internal-tooling project, so
expect rough edges — bug reports about them are useful.

## Getting set up

The backend and frontend have separate dependency sets. Use a virtual
environment for each, as described in the [README](README.md#running-it).

Because fine-tuning needs a model and real imagery, most changes are easiest to
test against a small GeoTIFF of your own. Please do not commit test rasters —
they are large, and `storage/` is git-ignored for that reason.

## Reporting a bug

Open an issue including:

- what you did, and what happened instead of what you expected
- your Python version and operating system
- whether you were running on CPU or GPU
- the full traceback, if there was one
- the approximate size and band count of the GeoTIFF, if it is raster-specific

Large-raster problems and coordinate-system problems are both common here, so
those details matter more than usual.

## Pull requests

1. Open an issue first for anything beyond a small fix, so we can agree on the
   approach before you spend time on it.
2. Branch from `master`.
3. Keep the change focused — one concern per pull request.
4. Describe how you tested it. There is no automated test suite yet, so say what
   you ran and what you saw.
5. Make sure nothing under `storage/`, no `__pycache__/`, and no model weights
   are in your diff.

## Style

- Standard Python conventions. Four-space indent, `snake_case`, type hints on
  new function signatures.
- Keep backend and frontend concerns separate. The backend should not assume a
  PyQt client; the frontend talks to it only through `utils/api_client.py`.
- Raise `HTTPException` with a useful status code and message rather than
  letting an exception escape a route.

## Things worth fixing

If you are looking for somewhere to start:

- **No tests.** There is no test suite at all. Unit tests for `bbox_utils` and
  the tile maths in `image_utils` would be the most valuable first contribution.
- **No authentication.** Every endpoint is open. Fine for a local tool,
  limiting for anything else.
- **Static and tile routes build paths from user input.** `/static/{path}` and
  `/tiles/{path}` join the request path onto a directory without normalising it.
  They should validate that the resolved path stays inside the intended
  directory.
- **CORS is `allow_origins=["*"]` with `allow_credentials=True`.** Browsers
  reject that combination, so the current setting does not do what it looks like
  it does. Pick an explicit origin list or drop the credentials flag.
- **Blocking work in request handlers.** Tiling a large raster happens inline
  and will hold a worker for the duration.
- **Training progress is coarse.** Percentages are hardcoded at a few
  checkpoints rather than reflecting real epoch progress.

## Dependency licensing

Keep this in mind before adding a dependency.

This repository is MIT. The backend's dependencies are permissive — FastAPI,
PyTorch, DeepForest, rasterio, Pillow and the rest are MIT, BSD or Apache-2.0,
all compatible.

The frontend is different. **PyQt5 is dual-licensed GPL v3 and commercial.**
Kesowa's own source stays MIT, but a redistributed binary or source bundle of
the frontend is a combined work subject to GPL v3 unless a commercial Riverbank
licence covers it. Using it internally is unaffected; distributing it is where
this bites. Porting the frontend to [PySide6](https://doc.qt.io/qtforpython/),
which is LGPL and largely API-compatible, would remove the constraint — that
would be a welcome contribution.

Please do not add GPL or AGPL dependencies to the backend.

## Secrets

Never commit credentials, API keys or tokens. Configuration belongs in
environment variables; `.env` is git-ignored. If you commit a secret by
accident, treat it as compromised and rotate it — deleting it in a later commit
does not remove it from history.

## License

Contributions are accepted under the [MIT License](LICENSE) covering this
repository.
