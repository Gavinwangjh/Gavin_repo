# Git Workflow

The project is prepared for Gavin's GitHub repository:

```text
https://github.com/Gavinwangjh/Gavin_repo.git
```

The delivery branch requested for this project is:

```text
ETL数据管道系统
```

## Before Committing

Run:

```bash
uv run ruff check src tests alembic
uv run pytest -p no:cacheprovider tests
```

Check changed files:

```bash
git status
git diff
```

## Commit

```bash
git add -A
git commit -m "Complete ETL data pipeline system"
```

## Push To Gavin Repository

If the `gavin` remote does not exist:

```bash
git remote add gavin https://github.com/Gavinwangjh/Gavin_repo.git
```

Push the current branch:

```bash
git push gavin HEAD:ETL数据管道系统
```

## Notes

- Do not commit `.env`.
- Commit `.env.example`, migrations, source files, tests, and docs.
- Keep generated cache directories out of Git.
- If a push is rejected because the remote branch already exists, pull or force-push only after confirming the remote branch can be replaced.
