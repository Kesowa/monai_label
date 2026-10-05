# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.

- Use GitHub's [private vulnerability reporting](https://github.com/Kesowa/monai_label/security/advisories/new)
  on this repository, or
- email **security@kesowa.com** with the details.

Include what you did, what happened, and what you expected, with enough detail
to reproduce. We will acknowledge your report and tell you whether we intend to
fix it, and when.

Please only test against your own installation.

## Threat model

Read this before reporting, because it determines whether a finding is a
vulnerability or the documented design.

This is a **local, single-user desktop tool**. The backend is intended to be
reachable only by the PyQt client on the same machine. By design it has:

- **no authentication or authorization** on any endpoint
- **no multi-tenancy** — there are no users, and all data is shared
- a bind address of `0.0.0.0:8000`, which exposes it to the local network

None of those three is a vulnerability on its own; they are consequences of the
intended deployment, and the README says not to expose port 8000. What we do
want to hear about is anything that breaks the tool's own boundaries, or that
makes a local-only assumption fail.

## In scope

- **Path traversal** in the file-serving routes. `/static/{path}` and
  `/tiles/{path}` build filesystem paths from request input, and reading files
  outside the intended directories is a real finding.
- **Arbitrary file write** through upload handling or annotation filenames.
- **Code execution** through a crafted GeoTIFF, model checkpoint or annotation
  file. Note that PyTorch checkpoint loading is a known risk area: a `.pt` file
  is untrusted input.
- **Unsafe deserialization** anywhere in the annotation or model-loading paths.
- **Vulnerable dependencies** in `backend/requirements.txt` or
  `frontend/requirements_pyqt.txt`.
- **Committed credentials** anywhere in the repository or its git history.
- **Denial of service** that a local user could trigger accidentally — for
  example an upload that exhausts memory rather than streaming.

## Out of scope

- The absence of authentication, as described in the threat model.
- The `0.0.0.0` bind address, which is documented.
- Anything requiring the operator to deliberately expose the service to an
  untrusted network against the README's guidance.
- Findings in DeepForest, PyTorch or other upstream dependencies — please
  report those to their maintainers, though we appreciate a heads-up.

## Model and data handling

This tool processes imagery you supply and writes it under
`backend/storage/`, unencrypted. Uploaded rasters can be geographically
sensitive. `storage/` is git-ignored, but it is your responsibility to protect
that directory and to avoid uploading imagery you are not permitted to hold.
