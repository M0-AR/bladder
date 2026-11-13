# Contributing to Bladder Vision

Thank you for improving open, reproducible medical imaging software. This project accepts code, documentation, datasets that can be shared lawfully, evaluation protocols, and careful criticism.

## The three rules that matter most

1. **Never commit patient data.** No videos, frames, annotations, reports, or metadata that could identify a person. Use synthetic or public data in examples.
2. **Never invent a clinical claim.** If you report a metric, include the dataset, split, threshold, sample size, and evaluation code. "Works well" is not a result.
3. **Keep the pipeline reproducible.** Seed randomness, record versions, and make commands copy-pasteable.

## Development setup

```bash
git clone https://github.com/M0-AR/bladder.git
cd bladder
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,cv]"
```

For model training or TFLite inference, also install the ML extra:

```bash
pip install -e ".[ml]"
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
pytest -q
```

All three must pass. Add tests for new pure-Python logic. For computer-vision changes, add a small synthetic test rather than a patient image.

## Code style

- Python 3.10+, type hints on public functions.
- Prefer small functions with explicit inputs and outputs.
- Keep heavy imports (`tensorflow`, `cv2`) inside the functions that need them when the module is otherwise lightweight.
- Do not add a dependency unless it removes more complexity than it adds.
- Public behaviour needs a docstring and, where relevant, an example in `README.md` or `docs/`.

## Commit messages

Use the imperative mood and explain the reason:

```text
Add seeded detection split with paired annotation checks
Fix threshold inversion in contour mask helper
Document the difference between Ta and NMIBC labels
```

## Data and privacy

- Public datasets: link to the source and its licence. Do not re-host data unless the licence allows it.
- Institutional data: keep it outside the repository. Describe the access process instead.
- Annotations: remove names, dates, procedure identifiers, and any burned-in overlays that could identify a patient.
- If you discover a privacy issue, use the private security advisory flow described in `SECURITY.md`.

## Clinical and regulatory language

This repository is research software. Contributions must not describe it as a diagnostic device, claim regulatory approval, or imply that a model is ready for patient care. See `DISCLAIMER.md` and `MODEL_FACTS.md`.

## Review checklist

- [ ] The change has a single clear purpose.
- [ ] Tests cover the new behaviour.
- [ ] Lint and formatting pass.
- [ ] Documentation is updated.
- [ ] No patient data or secrets are included.
- [ ] Clinical claims, if any, are sourced and scoped.
- [ ] New files have a compatible licence.

## Licence

By contributing, you agree that your contribution is licensed under the MIT License in `LICENSE`.