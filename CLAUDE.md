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

## Workflow

Code is written and edited **locally** using AI tools (Claude Code, GitHub Copilot) with the full benefit of file-based project structure, IDE features, and version control. Execution happens on **Kaggle's Jupyter server** to leverage free GPU resources.

The sync mechanism is GitHub: local changes are pushed to this repo, and the Kaggle kernel pulls from GitHub before each run. This means every push is immediately available for the next Kaggle session.

- `llm-memory.ipynb` is the Kaggle-side notebook — it lives on Kaggle and is not tracked here
- All other `.py` files, configs, and scripts are the local source of truth
- The remote is `git@github-kaggle:Tikhon-Radkevich/LLM-Memory.git` (uses a dedicated SSH key for Kaggle)
