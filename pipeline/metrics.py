"""
Purpose
-------
Collects and reports latency / throughput metrics for the voice pipeline.

Rationale
---------
A voice agent must be measured, not just felt. Every stage (VAD, ASR, LLM,
TTS, playback) contributes to the delay the user experiences. This module
timestamps each stage, tracks per-utterance end-to-end latency, and persists
results so that regressions can be caught and the system can be tuned.

Assumptions
-----------
- All timing uses time.perf_counter() (monotonic, ~100ns resolution on Windows).
- A single collector is shared across worker threads; every method is guarded
  by a lock so the audio callback is never blocked for long.
- Pipelines are identified by the integer id allocated by PipelineController.

Tradeoffs
---------
Keeping a transaction per utterance costs a little memory, which is bounded
by trimming finished transactions from the hot registry.
"""

import csv
import json
import statistics
import threading
import time
from collections import defaultdict, deque


class MetricsCollector:

    def __init__(self, max_transactions=500):
        # RLock so summary/to_json can re-enter while an outer call holds the lock.
        self._lock = threading.RLock()
        self._max_transactions = max_transactions

        # per-pipeline timing registry (integer pipeline id -> dict)
        self._tx = {}

        # global counters
        self.counters = defaultdict(int)

        # queue depth samples from the background sampler thread
        self.queue_samples = []

        # per-stage error counts
        self.errors = defaultdict(int)

        self._start_time = time.perf_counter()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start_pipeline(self, pipeline_id, eos_time):
        """Record the moment an utterance was captured (end of speech)."""
        with self._lock:
            self._tx[pipeline_id] = {
                "pipeline_id": pipeline_id,
                "eos_time": eos_time,
                "asr": None,
                "llm": None,
                "tts": None,
                "first_audio_time": None,
                "barge_in": False,
            }

    def end_pipeline(self, pipeline_id):
        """Freeze a completed pipeline and trim the registry to bounded size."""
        with self._lock:
            self._tx.pop(pipeline_id, None)
            if len(self._tx) > self._max_transactions:
                # drop the oldest transactions to keep memory bounded
                for oldest in list(self._tx.keys())[: len(self._tx) - self._max_transactions]:
                    self._tx.pop(oldest, None)

    # ------------------------------------------------------------------ #
    # stage timings
    # ------------------------------------------------------------------ #
    def record_asr(self, pipeline_id, t_start, t_end, language, chars):
        with self._lock:
            tx = self._tx.get(pipeline_id)
            if tx is None:
                return
            tx["asr"] = {
                "duration": t_end - t_start,
                "language": language,
                "chars": chars,
            }

    def record_llm(self, pipeline_id, t_first_token, t_end, tokens, chars):
        with self._lock:
            tx = self._tx.get(pipeline_id)
            if tx is None:
                return
            tx["llm"] = {
                "first_token_latency": t_first_token - tx["eos_time"],
                "generation_time": t_end - t_first_token,
                "tokens": tokens,
                "chars": chars,
            }

    def record_tts(self, pipeline_id, t_start, t_end, samples):
        with self._lock:
            tx = self._tx.get(pipeline_id)
            if tx is None:
                return
            if tx["tts"] is None:
                tx["tts"] = {
                    "first_synth_latency": t_start - tx["eos_time"],
                    "total_time": 0.0,
                    "samples": 0,
                    "sentences": 0,
                }
            tx["tts"]["total_time"] += t_end - t_start
            tx["tts"]["samples"] += samples
            tx["tts"]["sentences"] += 1

    def record_first_audio(self, pipeline_id, t):
        with self._lock:
            tx = self._tx.get(pipeline_id)
            if tx is None:
                return
            if tx["first_audio_time"] is None:
                tx["first_audio_time"] = t

    def mark_barge_in(self, pipeline_id):
        with self._lock:
            self.counters["barge_ins"] += 1
            tx = self._tx.get(pipeline_id)
            if tx is not None:
                tx["barge_in"] = True

    # ------------------------------------------------------------------ #
    # global counters
    # ------------------------------------------------------------------ #
    def count(self, name, n=1):
        with self._lock:
            self.counters[name] += n

    def record_error(self, stage):
        with self._lock:
            self.errors[stage] += 1

    def record_queue_sample(self, asr_size, llm_size, tts_size):
        with self._lock:
            self.queue_samples.append(
                (time.perf_counter(), asr_size, llm_size, tts_size)
            )

    # ------------------------------------------------------------------ #
    # aggregation + reporting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stats(values):
        if not values:
            return {"n": 0}
        values = sorted(values)
        return {
            "n": len(values),
            "min": round(min(values), 3),
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "p90": round(values[int(len(values) * 0.9) - 1], 3),
            "max": round(max(values), 3),
        }

    def _collect_values(self):
        """Extract flat lists of numeric metrics from all transactions."""
        asr_times, llm_first, llm_tps, tts_rtf = [], [], [], []
        e2e_latencies, synthesis_times = [], []
        durations = []
        sample_rates = []

        for tx in self._tx.values():
            eos = tx["eos_time"]
            if tx["asr"]:
                asr_times.append(tx["asr"]["duration"])
            if tx["llm"]:
                llm_first.append(tx["llm"]["first_token_latency"])
                gen = tx["llm"]["generation_time"]
                if gen > 0:
                    llm_tps.append(tx["llm"]["tokens"] / gen)
            if tx["tts"]:
                synthesis_times.append(tx["tts"]["total_time"])
                if tx["tts"]["total_time"] > 0 and tx["tts"]["samples"] > 0:
                    tts_rtf.append(tx["tts"]["total_time"] * 16000 / tx["tts"]["samples"])
            if tx["first_audio_time"] is not None:
                e2e_latencies.append(tx["first_audio_time"] - eos)

        return {
            "asr_time": asr_times,
            "llm_first_token": llm_first,
            "llm_tokens_per_sec": llm_tps,
            "tts_rtf": tts_rtf,
            "tts_total_time": synthesis_times,
            "e2e_latency": e2e_latencies,
        }

    def summary(self):
        """Return a printable summary dict of aggregate statistics."""
        with self._lock:
            values = self._collect_values()
            queues = [q[1:] for q in self.queue_samples]
            counters = dict(self.counters)
            errors = dict(self.errors)
            uptime = time.perf_counter() - self._start_time

        report = {
            "uptime_seconds": round(uptime, 1),
            "counters": counters,
            "errors": errors,
        }
        for name, vals in values.items():
            report[name] = self._stats(vals)

        if queues:
            per_pos = list(zip(*queues))
            report["queue_depths"] = {
                "asr": self._stats(per_pos[0]),
                "llm": self._stats(per_pos[1]),
                "tts": self._stats(per_pos[2]),
            }
        return report

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_csv(self, path):
        """Write per-utterance transactions to a CSV for offline analysis."""
        rows = []
        with self._lock:
            for tx in self._tx.values():
                eos = tx["eos_time"]
                row = {
                    "pipeline_id": tx["pipeline_id"],
                    "e2e_latency_ms": round((tx["first_audio_time"] - eos) * 1000, 1)
                    if tx["first_audio_time"] else "",
                    "asr_ms": round(tx["asr"]["duration"] * 1000, 1) if tx["asr"] else "",
                    "asr_language": tx["asr"]["language"] if tx["asr"] else "",
                    "asr_chars": tx["asr"]["chars"] if tx["asr"] else "",
                    "llm_first_token_ms": round(tx["llm"]["first_token_latency"] * 1000, 1)
                    if tx["llm"] else "",
                    "llm_tokens_per_sec": round(tx["llm"]["tokens"] / tx["llm"]["generation_time"], 1)
                    if tx["llm"] and tx["llm"]["generation_time"] > 0 else "",
                    "llm_tokens": tx["llm"]["tokens"] if tx["llm"] else "",
                    "tts_synth_ms": round(tx["tts"]["total_time"] * 1000, 1) if tx["tts"] else "",
                    "tts_sentences": tx["tts"]["sentences"] if tx["tts"] else "",
                    "barge_in": tx["barge_in"],
                }
                rows.append(row)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["pipeline_id"])
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def to_json(self, path):
        with self._lock:
            payload = {
                "summary": self.summary(),
                "transactions": list(self._tx.values()),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True

    def print_summary(self):
        report = self.summary()
        print("\n==================== PIPELINE METRICS ====================")
        print(f"Uptime: {report['uptime_seconds']}s | "
              f"Counters: {report['counters']} | Errors: {report['errors']}")
        for key in ("asr_time", "llm_first_token", "llm_tokens_per_sec",
                    "tts_rtf", "tts_total_time", "e2e_latency"):
            if report[key]["n"]:
                s = report[key]
                print(f"  {key:22s} n={s['n']:3d} mean={s['mean']:7.3f} "
                      f"p50={s['median']:7.3f} p90={s['p90']:7.3f} "
                      f"min={s['min']:7.3f} max={s['max']:7.3f}")
        if report.get("queue_depths"):
            for name, s in report["queue_depths"].items():
                if s["n"]:
                    print(f"  queue[{name}] n={s['n']:3d} mean={s['mean']:7.3f} max={s['max']:7.3f}")
        print("==========================================================\n")


# Singleton shared across the pipeline
metrics = MetricsCollector()
