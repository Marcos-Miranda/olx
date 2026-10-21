---
name: implement-track
description: Implement a track's plan task by task, with a review and a commit per block. Use when the user wants to implement or continue a track.
argument-hint: "[track_id]"
---

Track ID (optional): $ARGUMENTS

If no track ID was given, pick the `in_progress` track from `tracks/registry.md`, or else the first `not_started` one.

1. Mark the track `in_progress` in the registry if it isn't already.
2. Read `tracks/<track_id>/spec.md` and `tracks/<track_id>/plan.md`.
3. Work through the tasks not yet done (`[ ]` or `[~]`), in order. Mark each task `[~]` in plan.md before starting it and `[x]` once it's done. For a test task, write the test and confirm it fails before implementing.
4. When all tasks of a block are done, ask the user to review the block. Apply changes until accepted, then commit: `feat: finished block <n> of track <track_id>`.
5. When all blocks are done, run the final verification, fix what fails, and check each item that is met.
6. Ask the user for a final review. After approval, mark the track `completed` in the registry and commit: `feat: finished track <track_id>`.
