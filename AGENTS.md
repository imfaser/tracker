# AGENTS.md

## Stack

- Python 3.14, managed by **uv** (not pip). Always use `uv run` or `uv sync`.
- FastMCP MCP server for SAM3 image/video segmentation.
- PyTorch + torchvision sourced from Aliyun CUDA 13.2 mirror (configured in `pyproject.toml` `[tool.uv.sources]`).
- Transformers (HuggingFace) as the ML framework.
- Build backend: hatchling. Wheel package is `src/utils`.

## Commands

```bash
uv sync                          # install/sync all deps (including dev)
uv run ruff check .              # lint
uv run ruff check --fix .        # lint + auto-fix
uv run ruff format .             # format
uv run pyright                   # typecheck
uv run pytest                    # run all tests
uv run pytest tests/unit/        # run unit tests only
uv run pytest tests/integration/ # run integration tests only
uv run python main.py            # run MCP server (HTTP on :8000)
```

Ruff handles both linting and formatting (configured in `ruff.toml`).

## Project Layout

```
main.py                # MCP server entrypoint (FastMCP, HTTP transport)
src/utils/             # package root (utils.models, utils.tracker, utils.video)
tests/
  fixture/             # test resources — put hard-to-mock files here (model weights, sample images, etc.)
  integration/         # integration tests (require GPU / real model loading)
  unit/                # unit tests (isolated, no GPU needed)
pyproject.toml         # project config + uv sources
ruff.toml              # lint/format config
uv.lock                # lockfile — commit this
```

### Import path

`main.py` imports from `utils.*` (not `src.utils.*`). The hatch build config maps `src/utils` → `utils` in the wheel, so `uv run python main.py` resolves `from utils.models import ...` correctly. Do not add `src` to sys.path manually.

## Lint / Format Quirks

`ruff.toml` has ML-specific relaxations:
- `E501` ignored — line-length is 120 but not enforced (tensor shape defs get long).
- `UP045` disabled — use `Optional[X]` not `X | Y` (PyTorch/Transformers compat).
- `B008` disabled — allows function calls in default args (FastMCP `@mcp.tool()` pattern).
- `SIM108` disabled — if-else preferred over ternary for ML readability.
- `RUF001`/`RUF002`/`RUF003` ignored — Chinese characters allowed in source.

**Gotcha:** `known-first-party` in `ruff.toml` is still set to `my_mcp_server` — should be `utils`. Fix this if isort groups look wrong.

## Testing

- Framework: **pytest** (no conftest.py yet).
- Three test directories with distinct roles:
  - `tests/fixture/` — resources that are hard to mock (sample images, model configs, video clips). Not executable tests; pytest should skip or ignore non-`.py` files here.
  - `tests/unit/` — pure logic, no GPU required. Prefer these for fast feedback.
  - `tests/integration/` — require CUDA GPU and real model loading. Mark these with `@pytest.mark.integration` when added.
- No snapshot or special test plugins installed.

## OpenSpec Workflow

This repo uses OpenSpec for change management. Skills live in `.opencode/skills/`:
- `openspec-propose` → propose a new change
- `openspec-apply-change` → implement tasks from a change
- `openspec-update-change` → revise a change plan
- `openspec-sync-specs` → sync delta specs to main
- `openspec-archive-change` → archive completed changes
- `openspec-explore` → explore/clarify before proposing

Specs and change artifacts live in `openspec/`.

## Environment

- Platform: Windows (pwsh shell).
- PyTorch requires CUDA 13.2; GPU needed for model work and integration tests.
- `.venv/`, `docs/`, `.opencode/`, `openspec/` are gitignored.
