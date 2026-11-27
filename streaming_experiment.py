"""Reproducible streaming experiment: Count-Min Sketch and HyperLogLog."""
import bisect
import hashlib
import csv
from array import array
import json
import math
import random
import sys
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "clickstream.csv"
RESULT_PATH = HERE / "results.json"
N = 500_000
VOCAB = 25_000
ZIPF = 1.07
SEED = 20261123
MASK32 = 0xFFFFFFFF


def build_cdf():
    weights = [1.0 / (rank ** ZIPF) for rank in range(1, VOCAB + 1)]
    total = sum(weights)
    cdf, cumulative = [], 0.0
    for weight in weights:
        cumulative += weight / total
        cdf.append(cumulative)
    cdf[-1] = 1.0
    return cdf


def generate_stream():
    """Write a deterministic, Zipfian clickstream to disk."""
    rng = random.Random(SEED)
    cdf = build_cdf()
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("event_id", "product_id"))
        for event_id in range(1, N + 1):
            rank = bisect.bisect_left(cdf, rng.random()) + 1
            writer.writerow((event_id, f"P{rank:05d}"))


def mix32(value: int) -> int:
    value &= MASK32
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & MASK32
    value = ((value ^ (value >> 15)) * 0x846CA68B) & MASK32
    return (value ^ (value >> 16)) & MASK32


def product_number(product_id: str) -> int:
    return int(product_id[1:])


class CountMinSketch:
    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth
        self.rows = [array("I", [0]) * width for _ in range(depth)]

    def _index(self, item_number: int, row: int) -> int:
        seed = ((row + 1) * 0x9E3779B9) & MASK32
        return mix32(item_number + seed) % self.width

    def add(self, product_id: str):
        item_number = product_number(product_id)
        for row in range(self.depth):
            self.rows[row][self._index(item_number, row)] += 1

    def estimate(self, product_id: str) -> int:
        item_number = product_number(product_id)
        return min(self.rows[row][self._index(item_number, row)]
                   for row in range(self.depth))

    @property
    def bytes(self):
        # Python integer/list overhead is runtime-dependent; this reports the
        # logical 32-bit counter-array equivalent for a fair fixed-size metric.
        return self.width * self.depth * 4


class HyperLogLog:
    def __init__(self, p: int):
        self.p = p
        self.m = 1 << p
        self.registers = bytearray(self.m)

    def add(self, product_id: str):
        hash_value = int.from_bytes(hashlib.blake2b(
            product_id.encode("utf-8"), digest_size=8, person=b"stream-hll").digest(), "big")
        index = hash_value >> (64 - self.p)
        remainder = hash_value & ((1 << (64 - self.p)) - 1)
        rank = 64 - self.p - remainder.bit_length() + 1
        if rank > self.registers[index]:
            self.registers[index] = rank

    def count(self) -> float:
        harmonic_sum = sum(2.0 ** (-register) for register in self.registers)
        zero_count = self.registers.count(0)
        if self.m == 16:
            alpha = 0.673
        elif self.m == 32:
            alpha = 0.697
        elif self.m == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1 + 1.079 / self.m)
        estimate = alpha * self.m * self.m / harmonic_sum
        if estimate <= 2.5 * self.m and zero_count:
            estimate = self.m * math.log(self.m / zero_count)
        two32 = 2 ** 32
        if estimate > two32 / 30:
            estimate = -two32 * math.log(1 - estimate / two32)
        return estimate

    @property
    def bytes(self):
        return self.m  # bytearray: one byte per register


def for_each_event(callback):
    """Read and process one CSV row at a time."""
    with CSV_PATH.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row in reader:
            callback(row[1])


def time_stream(callback):
    start = time.perf_counter()
    for_each_event(callback)
    return (time.perf_counter() - start) * 1000


def main():
    generate_stream()

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    exact_counts = {}
    exact_ms = time_stream(lambda item: exact_counts.__setitem__(
        item, exact_counts.get(item, 0) + 1))
    exact_heap_bytes = tracemalloc.get_traced_memory()[0] - before_current
    tracemalloc.stop()

    exact_distinct = len(exact_counts)
    top_100 = [item for item, _ in sorted(
        exact_counts.items(), key=lambda pair: pair[1], reverse=True)[:100]]
    cms_results = []
    for width, depth in [(256, 3), (512, 4), (1024, 5), (2048, 5)]:
        cms = CountMinSketch(width, depth)
        elapsed_ms = time_stream(cms.add)
        absolute_errors, relative_errors, overestimates = [], [], []
        for item, truth in exact_counts.items():
            estimate = cms.estimate(item)
            absolute_errors.append(abs(estimate - truth))
            relative_errors.append(abs(estimate - truth) / truth)
            overestimates.append(estimate - truth)
        top_abs = [abs(cms.estimate(item) - exact_counts[item]) for item in top_100]
        top_rel = [abs(cms.estimate(item) - exact_counts[item]) / exact_counts[item]
                   for item in top_100]
        cms_results.append({
            "width": width, "depth": depth, "bytes": cms.bytes,
            "elapsedMs": elapsed_ms, "throughput": N / (elapsed_ms / 1000),
            "meanAbsoluteError": sum(absolute_errors) / len(absolute_errors),
            "meanRelativeErrorPct": sum(relative_errors) / len(relative_errors) * 100,
            "maxOverestimate": max(overestimates),
            "top100MeanAbsoluteError": sum(top_abs) / len(top_abs),
            "top100MeanRelativeErrorPct": sum(top_rel) / len(top_rel) * 100,
        })

    hll_results = []
    for p in [8, 10, 12, 14]:
        hll = HyperLogLog(p)
        elapsed_ms = time_stream(hll.add)
        estimate = hll.count()
        hll_results.append({
            "p": p, "buckets": hll.m, "bytes": hll.bytes,
            "elapsedMs": elapsed_ms, "throughput": N / (elapsed_ms / 1000),
            "estimate": estimate, "exact": exact_distinct,
            "absoluteError": abs(estimate - exact_distinct),
            "relativeErrorPct": abs(estimate - exact_distinct) / exact_distinct * 100,
            "theoreticalRsePct": 1.04 / math.sqrt(hll.m) * 100,
        })

    result = {
        "dataset": {
            "name": "Synthetic Zipfian e-commerce clickstream", "records": N,
            "vocabulary": VOCAB, "zipfExponent": ZIPF, "seed": SEED,
            "fileBytes": CSV_PATH.stat().st_size,
            "note": "Deterministically generated, one CSV row per event; product popularity follows a Zipf distribution.",
        },
        "environment": {"runtime": sys.version.split()[0], "platform": sys.platform,
                        "arch": "x64" if sys.maxsize > 2**32 else "x86"},
        "groundTruth": {
            "distinctProducts": exact_distinct,
            "exactCountingElapsedMs": exact_ms,
            "exactMapHeapBytes": exact_heap_bytes,
        },
        "countMin": cms_results,
        "hyperLogLog": hll_results,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
