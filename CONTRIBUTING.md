# Contributing to AutoTraj

Thank you for your interest in contributing to AutoTraj!

---

## Setting Up the Dev Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/Thornwell16/gbtm_project.git
   cd gbtm_project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

3. **Install runtime and dev dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Verify the setup** by running the fast test suite:
   ```bash
   make test
   ```
   All tests should pass in under 30 seconds on a modern laptop.

---

## Running Tests Locally

| Command | What it does |
|---------|-------------|
| `make test` | Fast suite — skips `@pytest.mark.slow` tests |
| `make benchmark` | Cambridge benchmark tests only |
| `make coverage` | Fast suite + HTML coverage report in `htmlcov/` |
| `pytest` | Full suite including slow autosearch tests (~10 min) |
| `pytest -k test_name` | Run a single named test |
| `pytest -m recovery` | Run only parameter-recovery tests |

The `cambridge.txt` dataset must be present in the project root for the
benchmark suite. It is included in the repository.

---

## Adding New Tests

All tests live in the `tests/` directory. The three existing suites are:

- `test_parameter_recovery.py` — simulation-based ground-truth recovery
- `test_cambridge_benchmark.py` — published dataset validation
- `test_edge_cases.py` — pathological inputs and boundary conditions

**Guidelines for new tests:**

- Use `@pytest.fixture(scope="module")` for expensive model fits that are
  shared by multiple tests in the same file. This avoids refitting the same
  model multiple times.
- Mark slow tests (anything fitting more than ~5 models or using `n_starts > 5`)
  with `@pytest.mark.slow` so they are excluded from the fast CI run.
- Assert on meaningful quantities — log-likelihood, group proportions, SE
  ratios — rather than exact parameter values, which can be sensitive to
  random initialisation.
- Provide a descriptive failure message in every `assert`:
  ```python
  assert value > threshold, f"Expected > {threshold}, got {value:.4f}"
  ```
- Place data-generation helpers (`simulate_*` functions) at the top of the
  test file, not inside the test functions, so fixtures can reuse them.

---

## Code Style Guidelines

AutoTraj follows a pragmatic subset of PEP 8, enforced by **flake8**:

- **Line length:** 120 characters max (`max-line-length = 120` in `.flake8`).
- **Naming:** `snake_case` for functions and variables; `UPPER_CASE` for
  module-level constants.
- **Imports:** Standard library first, then third-party (`numpy`, `scipy`,
  `numba`, `pandas`), then local imports. One blank line between groups.
- **Docstrings:** Google style for all public functions:
  ```python
  def my_function(x, y):
      """One-line summary.

      Longer description if needed.

      Args:
          x: Description of x.
          y: Description of y.

      Returns:
          Description of return value.
      """
  ```
- **Numba JIT functions** (`@njit`): Comments explaining the mathematical
  operation are required inside JIT kernels because type-inference errors
  can be cryptic. Reference the formula (e.g., `# ∂log p / ∂μ`) wherever
  a gradient is computed.
- **No logic in docstrings or comments.** Documentation describes intent;
  code implements it.

Run `make lint` before opening a pull request — the CI will fail if flake8
reports any errors.

---

## Pull Request Process

1. Create a feature branch off `main`:
   ```bash
   git checkout -b feature/my-change
   ```
2. Make your changes, add tests if appropriate, and run `make test lint`.
3. Open a pull request against `main`. The CI will automatically run tests
   and lint checks.
4. Describe what the PR changes and why. Link to any relevant issues.

---

## Questions

Open an issue on GitHub for bug reports, feature requests, or questions about
the statistical methodology.
