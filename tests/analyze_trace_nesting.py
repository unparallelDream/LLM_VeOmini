import collections
import gzip
import json
import sys


with gzip.open(sys.argv[1], "rt", encoding="utf-8", errors="replace") as handle:
    events = json.load(handle, strict=False)["traceEvents"]

cpu = [
    event
    for event in events
    if event.get("ph") == "X" and event.get("cat") == "cpu_op" and event.get("dur")
]
by_thread = collections.defaultdict(list)
for event in cpu:
    event["_end"] = event["ts"] + event["dur"]
    by_thread[(event.get("pid"), event.get("tid"))].append(event)


parent_chains = {}
for thread_events in by_thread.values():
    stack = []
    for event in sorted(thread_events, key=lambda item: (item["ts"], -item["dur"])):
        while stack and stack[-1]["_end"] < event["_end"]:
            stack.pop()
        parent_chains[id(event)] = tuple(parent.get("name") for parent in reversed(stack[-8:]))
        stack.append(event)


for target_name in ("aten::nonzero", "aten::index"):
    targets = [event for event in cpu if event.get("name") == target_name]
    print(f"{target_name} count={len(targets)}")
    signatures = collections.Counter()
    for event in targets:
        signatures[parent_chains[id(event)]] += 1
    for chain, count in signatures.most_common(12):
        print(count, " <- ".join(chain))

mass = [event for event in cpu if event.get("name") == "_MassEnvelopeTopKV2FnBackward"]
print(f"mass_backward count={len(mass)}")
if mass:
    sample = mass[0]
    nested = [
        event
        for event in by_thread[(sample.get("pid"), sample.get("tid"))]
        if event is not sample and sample["ts"] <= event["ts"] and sample["_end"] >= event["_end"]
    ]
    counts = collections.Counter(event.get("name") for event in nested)
    durations = collections.Counter()
    for event in nested:
        durations[event.get("name")] += event["dur"]
    for name, count in counts.most_common(40):
        print(f"{count:4d} {durations[name] / 1000:10.3f} ms {name}")

generic = sorted(
    float(event["dur"]) / 1000.0
    for event in events
    if event.get("ph") == "X" and event.get("cat") == "kernel" and event.get("name") == "kernel_kernel"
)
if generic:
    print(f"kernel_kernel count={len(generic)} total_ms={sum(generic):.3f}")
    bins = ((0, 1), (1, 5), (5, 20), (20, 50), (50, 100), (100, 1000))
    for low, high in bins:
        values = [value for value in generic if low <= value < high]
        print(f"[{low},{high})ms count={len(values)} total_ms={sum(values):.3f}")
    print("largest_ms", " ".join(f"{value:.3f}" for value in generic[-50:]))
