# Contributing to Mnemosyne

Thank you for your interest in contributing.

## Setup

```bash
git clone https://github.com/castnettech/mnemosyne.git
cd mnemosyne
pip install -e ".[dev]"
pytest mnemosyne/tests/
```

## Guidelines

- **Zero runtime dependencies.** Contributions adding runtime deps will not be accepted. The stdlib-only constraint is a core design decision, not a temporary limitation.
- **Tests required.** Every PR must include tests. The baseline is 293+ tests — don't lower it.
- **One concern per PR.** Keep changes focused. A bug fix is not a refactor opportunity.
- **Run the full suite** before submitting: `pytest mnemosyne/tests/ -q`

## Code Style

- Python 3.11+ type annotations
- No docstring is better than a wrong docstring
- Follow existing patterns — read neighboring code before writing new code

## Reporting Issues

Open an issue at https://github.com/castnettech/mnemosyne/issues with:
- What you expected
- What happened
- Steps to reproduce
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). You also agree to the Contributor License Agreement (CLA), which grants the project owner the right to offer your contributions under alternative licenses. The CLA is signed once via GitHub as part of your first pull request.
