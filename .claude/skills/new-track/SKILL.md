---
name: new-track
description: Create a spec-driven development track (spec.md and plan.md) for a task. Use when the user wants to start a new feature or task.
argument-hint: <task description>
---

Task: $ARGUMENTS

1. Read the relevant code and ask the user about anything unclear. Don't write files until the task is clear.
2. Create the track ID `<word1>_<word2>_<YYYYMMDD>`: two words naming the task and today's date (`date +%Y%m%d`). Create the folder `tracks/<track_id>/`.
3. Write `tracks/<track_id>/spec.md`:
   ```markdown
   # Spec: <track_id>

   ## Functional requirements
   - FR1: ...

   ## Acceptance criteria
   - AC1: ...
   ```
4. Ask the user to review the spec: accept or suggest changes. Repeat until accepted.
5. Write `tracks/<track_id>/plan.md` with exactly what to implement to meet the spec. Write it for a human reviewer: describe modules, functions and behavior, with no code beyond short signatures. Group tasks into coherent blocks. Every task has a checkbox, and a test task (when needed) comes before the implementation it covers (TDD):
   ```markdown
   # Plan: <track_id>

   ## Block 1: <name>
   - [ ] Test: <behavior under test>
   - [ ] <implementation task>

   ## Final verification
   - [ ] `uv run ruff check` and `uv run ruff format --check` pass
   - [ ] `uv run mypy .` passes
   - [ ] `uv run pytest` passes
   - [ ] AC1: ...
   ```
6. Ask the user to review the plan: accept or suggest changes. Repeat until accepted.
7. Add the track to `tracks/registry.md` with status `not_started`, then commit the track folder and the registry: `feat: started track <track_id>`.
