# Workflow

## One Iteration Cycle

1. Read the iteration goal.
2. Update only the files required for that iteration.
3. Run the stated check command.
4. Fix failures before moving to the next iteration.
5. Update `doc/tasklist.md` when the result is stable.

## Before Code

Confirm the input file, expected output file, and verification command. If a generated artifact can be rebuilt, do not treat it as the source of truth.

## Checks

Use focused checks first, then run the full test suite:

```bash
python scripts/build_index.py
pytest tests -v
```

## Commits

Commit only after the pipeline and tests pass. Keep generated binary index artifacts out of git.

## Forbidden Actions

- Do not answer from outside retrieved context.
- Do not silently ignore missing data.
- Do not hide low retrieval scores.
- Do not make unrelated changes to the RideFlow forecasting project.
