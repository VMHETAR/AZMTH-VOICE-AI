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
