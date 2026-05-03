# LLM-Memory Project

## Auto-commit workflow

After every code change, commit and push all modifications to the `main` branch immediately.

**Rules:**
- Stage all relevant changed files (avoid `.env`, secrets, credentials, `llm-memory.ipynb`)
- Commit to `main` branch directly
- Push to `origin main` after every commit
- Use short, concise commit messages that describe what changed (e.g. `add data loader`, `fix preprocessing bug`, `update model config`)
- No need to ask for confirmation before committing and pushing — just do it

**Command pattern:**
```
git add <changed files>
git commit -m "short description"
git push origin main
```

## Context

This project runs on Kaggle GPU via GitHub sync. Code is edited locally with AI tools (Claude Code, Copilot) and synced to Kaggle kernels through GitHub. Fast commit/push cycle is critical.
