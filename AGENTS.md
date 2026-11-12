# AGENTS.md

Operating contract for agents working in the nanodit repo.

## What this repo is

A from-scratch text-to-image DiT on one consumer GPU. `docs/SPEC.md` is the design, `docs/HANDOVER.md` says where
things stand, `README.md` has the layout and how to run it.

## Who decides what

The maintainer decides the design: anything SPEC states, which questions are open or closed, what gets merged into
the design. Agents write the code, run the experiments, keep the record and commit.

- Implement what SPEC says. A deviation, however small, is a design change: stop and bring it with the facts that
  decide it. Do not improve the design silently, and do not write an experiment's winner into SPEC yourself.
- An observation is not a fix request. Report it; change nothing until asked.
- Verify the risky assumption before building on it, with a small script, and report the numbers.
- Never discard a working state without isolating the cause of a problem.
- Measurement beats the docs. When a finding contradicts something written, correct it where the wrong claim lives.

## Documentation

Each fact has one home. Before writing, find where it belongs; if it is already written, change it there.

| File | Holds | Rule |
|---|---|---|
| `README.md` | what the project is, layout, how to run | for a reader who has never seen the repo |
| `docs/SPEC.md` | the settled design, present tense | no history, no reasons, no numbers from runs |
| `docs/PLAN.md` | open questions, grouped | a question, how it would be settled, the state in a few lines; nothing closed, no diary |
| `docs/FACTS.md` | measured numbers that the design or an open question rests on | each with source and date; replaced when superseded, never appended to |
| `docs/log/` | one note per run or read, `YYYY-MM-DD-title.md` | full tables; written once |
| `docs/topics/` | one doc per side quest: its question, design, reads and numbers | PLAN keeps the heading, one line of state and the link |
| `docs/archive/` | closed questions, one note per closure, `YYYY-MM-DD-title.md` | read for the reasons; never edited |
| `docs/HANDOVER.md` | where things stand, what waits on a decision, traps | rewritten each time, never appended to |
| `docs/REFERENCES.md`, `docs/references/` | sources and what each establishes; verbatim source material | |

How a question moves: found, it goes into PLAN. Measured, the numbers go to a log note, and the one line that matters
to FACTS. Grown past a screen, it becomes a topic. Closed by the maintainer, it leaves PLAN in the same session, its
conclusion and reason go into a new archive note, and SPEC gets the result and only the result.

How to write them:

- Impersonal and in the present tense. No names, no "I", no account of who said what or when a session happened.
  What matters about a statement is its status, so mark that: *Decided*, *Measured*, *Hypothesis*, *Not wanted*.
  A date belongs on a measurement and on a decision, nowhere else.
- A number cites a script in `scripts/` or a run directory. A result from a script that was not kept is not a fact;
  commit the script or do not record the number.
- One seed is one seed. Say how large a difference the comparison can resolve before reading a smaller one.
- Do not narrate. A log note has what ran, the table and the read; it does not have the order things were tried in.
- When code or a config is removed, fix the references to it in the same commit.

## Code

- Every line earns its place against the task. No flags, branches, adapters, warnings or "just in case" scaffolding.
  When the shape is wrong, prefer deletion or a sharper boundary over accommodation.
- A switch exists for an open question in PLAN. When the question closes, the losing branch of the switch is deleted
  in the same change. The trunk carries the design plus the switches of questions still open, nothing else.
- One implementation of each thing. No parallel compiled and uncompiled paths, no second copy of a model
  construction or an encoder call in another script.
- When a data layout changes, the data is migrated and the old layout is removed from the code. No side files, no
  reader that accepts both.
- Code holds no machine paths; scripts take them as arguments. Configs may hold them.
- Pure PyTorch. No HF Trainer, no Accelerate, no ai-toolkit.
- Boring and clear over clever; explicit domain concepts over generic plumbing. Comments for non-obvious maths,
  tensor-shape invariants and performance traps, never to narrate an operation.
- EmbeddingGemma 2 runs in bf16 or fp32, never fp16.
- Tests test behaviour. Change a test when the design changes, not to get back to green.

## Experiments

- `main` is the trunk: what SPEC describes, and where every run that counts starts from.
- An arm that changes code lives on a branch `arm/<name>`, committed before it runs. Never run from uncommitted
  code: the commit is the run's only description. An arm that is only a config runs from `main`.
- Work in the one checkout. No extra worktrees left behind; if one is needed to run an arm while the trunk moves,
  remove it when the run ends.
- When an arm is read, write its log note, then either bring it to the maintainer for merging or leave the branch as
  the record. Delete its config from `configs/` unless it is a reference other arms compare against; the run
  directory keeps the `config.toml` it ran with.
- `configs/` holds only configs that load on the current code and data.
- Checkpoints: an arm keeps none once it is scored, unless something continues from it. A long run keeps its last.
  A checkpoint that no longer loads into the model is deleted.
- Logs are JSONL in the run directory, one object per line; wandb mirrors them (SPEC §8). The JSONL is the record.

## Commits

Agents own commits. A subagent reviews each unit against SPEC and this file; the session judges the review and
commits. One commit per coherent change: a design change, a probe and a doc correction are separate commits. The
subject says what changed; a body when intent or impact needs it. Pushing stays with the maintainer.

## Running things

`uv sync` from the repo root. torch 2.15 comes from the PyTorch cu132 index. usuitrack is an editable path dependency
on the sibling checkout `../usuitrack-zero-init`, so it runs whatever is checked out there.

Scripts run from the repo root as `uv run python scripts/<name>.py ARGS`; the usage is in each docstring. They import
each other by module name (`from train import Config`).

A run streams to wandb (`wandb_project` in its config) with the entity and key from `$WANDB_ENTITY` and
`$WANDB_API_KEY`; `WANDB_MODE=offline` runs without the network. A run directory that exists is refused; one whose
`wandb.init` failed has to be deleted before rerunning.

The data lives outside the repo, in the directories the configs name. On the development machine these still carry
the project's first name, `tiny-dit` (archive, 2026-11-11).

The GPU (RTX 4070 Super, 12 GB) is shared with other jobs. Check `nvidia-smi` before launching, and launch training
in the background, never in the foreground.

## Validation

```bash
uv run python -m pytest tests/
```

Work is not done because imports resolve or a smoke run descends. If validation could not run, say exactly what is
unverified. A report says what was verified, what was assumed, and what is still open.
