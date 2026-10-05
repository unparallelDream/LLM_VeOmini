import collections
import gzip
import json
import sys


path = sys.argv[1]
with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
    trace = json.load(handle, strict=False)
events = trace["traceEvents"]
category_counts = collections.Counter((event.get("cat"), event.get("ph")) for event in events)
print("CATEGORIES")
for key, count in category_counts.most_common(20):
    print(key, count)


def report(title, predicate, limit=50):
    totals = collections.Counter()
    calls = collections.Counter()
    for event in events:
        if event.get("ph") == "X" and predicate(event):
            name = event.get("name", "<unnamed>")
            totals[name] += float(event.get("dur", 0.0))
            calls[name] += 1
    grand = sum(totals.values())
    print(f"\n{title} total_us={grand:.3f} unique={len(totals)}")
    for name, duration in totals.most_common(limit):
        share = 100.0 * duration / grand if grand else 0.0
        print(f"{duration / 1000.0:11.3f} ms {share:6.2f}% {calls[name]:6d} {name}")


report("CUDA_KERNELS", lambda event: event.get("cat") == "kernel")
report("CUDA_RUNTIME", lambda event: event.get("cat") == "cuda_runtime", limit=30)
report("CPU_OPS", lambda event: event.get("cat") == "cpu_op", limit=50)
