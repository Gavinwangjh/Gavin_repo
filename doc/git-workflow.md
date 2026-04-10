This guide explains how to work with Git in this project. The `main` branch is protected — you cannot push directly to it. All changes go through feature branches and pull requests.

---
## The Branching Strategy
- **main:** The "Holy Grail." This branch contains stable code. Direct pushes are strictly forbidden.

- **development:** The integration branch. This is the default branch. All feature branches start from here and merge back into here.

- **feature/*** or **fix/***: Temporary branches where actual work happens.

---
## Daily workflow
### 1. Always start from an updated main

Before starting any new work, make sure your local **development** branch is up to date:
```bash
git checkout development
git pull origin development
```

### 2. Create a feature branch

Create a new branch for each piece of work. Name it clearly:

```bash
git checkout -b feature/fetch-documents
```

Naming convention:

| Prefix | When to use |
|---|---|
| `feature/` | Adding new functionality |
| `fix/` | Fixing a bug |
| `chore/` | Non-code changes (docs, config) |

Examples:
```bash
feature/document-scraper
feature/temporal-workflow-parsing
fix/database-connection-timeout
chore/update-readme
```

### 3. Write your code

As you write code, use the provided tools to keep it clean:

- make format: Automatically fixes simple linting issues and formats your code (indentation, line lengths).

- make lint: Runs a full check (Ruff, Pyright, Secret detection) to ensure everything is perfect.

### 4. Commit regularly

Commit small, logical pieces of work. Do not accumulate everything in one massive commit.

```bash
git add .
git commit -m "add fetch_document activity"
```

Write commit messages that describe what changed, not how:

```
# Good
add fetch_document activity
fix retry logic for failed downloads
update database model for DocumentSegments

# Bad
changes
fix stuff
wip
```

### 5. Push your branch

```bash
git push origin feature/fetch-documents
```

### 6. Open a pull request

Go to the GitHub repository. Open a PR to merge your branch into **development** (NOT into **main**).

Fill in a description of your changes and wait for a review or CI status checks.

### 7. After merge

Once your pull request is merged, switch back to development and pull the latest:

```bash
git checkout development
git pull origin development
```

Delete your local branch if you no longer need it:

```bash
git branch -d feature/fetch-documents
```

---

## Pre-commit hooks

This project uses **pre-commit hooks** to enforce code quality. They act as a gatekeeper.

**What do they check?**
- Ruff: Syntax errors, unused imports, and the 110/120 character line limit.
- Secret Detection: Prevents accidental commits of API keys or passwords.
- Protection: A local hook prevents you from accidentally pushing directly to main.

```bash
make lint # Run all checks on all files
make format # Run only the auto-formatter/fixer
```

If pre-commit hooks are not running (e.g., after a fresh clone), install them:

```bash
uv run pre-commit install
```

This is done automatically by `make setup`.

---

## Common situations

### I made changes on main by mistake

Move your changes to a new branch without losing them:

```bash
git stash
git checkout -b feature/my-feature
git stash pop
```

### I need to update my branch with new changes from main

```bash
git checkout development
git pull origin development
git checkout feature/my-feature
git rebase development
```
### I want to undo my last commit (but keep the changes)
```Bash
git reset --soft HEAD~1
```

### I want to see what changed before committing
```Bash
git status   # see which files are modified
git diff     # see the actual line-by-line changes
```

### If there are conflicts, 
Git will tell you which files to resolve. Open them, fix the conflicts, then:

```bash
git add .
git rebase --continue
```

### I want to see what changed before committing

```bash
git diff
git status
```