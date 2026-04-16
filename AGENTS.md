# AGENTS.md

- Read `README.md` first for repository context. Read files under `codex_md/` only when you need project-specific harness guidance.
- Keep all ongoing work logs and history notes under `codex_history/`. Do not create new history markdown files outside that directory.
- When recording work history, include what was tried, why it was tried, what failed, why it failed, what succeeded, why it succeeded, and the next recommended step.
- Keep history notes detailed enough that a later Codex run can resume the investigation without redoing the same dead ends.
- Prefer `rg` / `rg --files` for search. Batch file reads when possible instead of reading files one by one.
- Use `apply_patch` for manual code edits. Do not overwrite files with ad-hoc scripts when a direct patch is sufficient.
- Treat `src/alpamayo1_5/` as product code, `tools/` as task automation, and `notebooks/` as exploratory artifacts. Avoid editing notebooks unless the task explicitly targets them.
- Do not edit generated caches or virtualenv contents under `a1_5_venv/`, `.git/`, or `tools/__pycache__/`.
- For TensorRT-LLM work, prefer validating the local installed API before assuming upstream examples match this environment.
- If a task involves Alpamayo inference/runtime changes, inspect `src/alpamayo1_5/models/` and `tools/` before proposing architecture changes.
- Leave rollback points in git whenever a meaningful step lands: make a coherent commit at each checkpoint, and prefer pushing it to `https://github.com/ys0333/alpamayo/tree/codex/trtllm-progress` so the work can be restored at any time.
- After Python code changes, run a targeted validation step when feasible, at minimum `python -m py_compile` for changed scripts or modules.
- Keep final responses short, state what changed, and mention any validation run or blocker explicitly.
