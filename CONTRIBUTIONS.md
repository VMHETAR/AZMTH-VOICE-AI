# Docstring Guidelines

## Philosophy

Docstrings should explain **why** a component exists, **what responsibility it owns**, and **what guarantees it provides**—not narrate how the code works line by line.

The implementation is already visible in the code. The docstring should provide the context that future contributors cannot infer by simply reading the implementation.

---

## Questions every docstring should answer

* Why does this component exist?
* What problem does it solve?
* Why was this design chosen?
* What assumptions does it make?
* What guarantees or invariants should remain true?
* Are there any important tradeoffs?

If a sentence merely describes what each line of code is doing, it probably does not belong in the docstring.

---

## Recommended Template

```python
"""
Purpose
-------
A short description of the responsibility of this component.

Rationale
---------
Why this component exists and why this design was chosen.

Assumptions
-----------
List any assumptions, invariants, or expectations.

Tradeoffs
---------
Mention any intentional compromises or design decisions.
"""
```

---

## Good Example

```python
"""
Purpose
-------
Provides real-time audio recording for the speech pipeline.

Rationale
---------
Recording is intentionally separated from speech detection so that
audio capture remains independent of downstream processing. This makes
the recorder reusable with different VAD and ASR implementations.

Assumptions
-----------
- Audio is captured continuously.
- Consumers process frames from the shared queue.
- No preprocessing is performed here.

Tradeoffs
---------
Continuous recording uses more memory than event-driven recording but
keeps latency low and simplifies synchronization.
"""
```

---

## Bad Example

```python
"""
Reads microphone input.

Starts a thread.

Adds audio to a queue.

Stops when requested.
"""
```
The code already communicates these implementation details. The docstring should instead explain the reasoning behind the design.

---

## Comments

Use inline comments sparingly. Comments should explain **why** something is done, not repeat what the code already expresses.

Good:

```python
# Keep one extra frame to avoid cutting off the beginning of speech.
```
Bad:
```python
# Increment i.
i += 1
```

# Git Workflow

## Branching

* Do **not** commit directly to `main`.
* Create a feature branch for every change.

Naming convention:

```
feature/<short-description>
fix/<short-description>
refactor/<short-description>
docs/<short-description>
test/<short-description>
```

Examples:

```
feature/realtime-vad
fix/audio-buffer-overflow
refactor/tts-pipeline
docs/update-contributing
```

---

## Commit Messages

Commits should describe **why** the change was made, not simply what files changed.

Format:

```
<type>: <short summary>
```

Types:

```
feat      New functionality
fix       Bug fix
refactor  Internal code improvement
docs      Documentation changes
test      Tests
perf      Performance improvements
chore     Maintenance tasks
```

Examples:

```
feat: add streaming speech segmentation

fix: prevent recorder thread from deadlocking

refactor: decouple VAD from audio capture

docs: document module docstring philosophy

perf: reduce audio queue allocations
```

Avoid commit messages like:

```
update
changes
final
done
fix
asdf
```

---

## Before Pushing

Ensure the following:

* Code runs without errors.
* Existing functionality is not broken.
* Documentation is updated if behavior changed.
* Remove debugging code (`print`, temporary logs, commented-out code).
* Format code consistently.

---

## Pull Requests

A pull request should clearly answer:

1. **What problem does this solve?**
2. **Why was this approach chosen?**
3. **Any important tradeoffs?**
4. **Anything reviewers should pay attention to?**

Small, focused pull requests are preferred over large unrelated changes.

---

## General Philosophy

* Make each commit represent a logical unit of work.
* Keep commits small and reviewable.
* Write code for the next developer—not just for the compiler.
* Optimize for readability before cleverness.
* Every change should have a clear reason.
