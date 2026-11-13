# CPU benchmark methodology

Measurements are in [cpu-benchmark.json](evidence/cpu-benchmark.json) and bounded
candidate decisions in [cpu-tuning.json](evidence/cpu-tuning.json), linked by the
[results manifest](results-manifest.json). These are **D32 smoke** measurements,
not D256 released-model throughput or a GPU speed claim.

The benchmark and trainer share model construction, data loaders, actual objective,
sample-weighted accumulation, clipping and AdamW updates. Each measurement uses
one CPU PyTorch thread, two warmup groups and ten timed groups. CUDA synchronization
and allocated/reserved-memory counters are implemented but unverified here.

| Scope | Included in timed region | Excluded |
| --- | --- | --- |
| loader | Fetch/generate each microbatch in an accumulation group | Model computation |
| step | Cached CPU group, transfer, forward/loss/backward, clip, AdamW | Loading a new group |
| end_to_end | Fetch a new group plus the same complete optimizer update | Epoch validation, checkpoint, MLflow |

Startup/model creation and warmup are recorded separately. The cache condition is
seeded synthetic generation; no real disk dataset was benchmarked. Loader wait,
actual sample counts (including partial final groups), update counts, group duration,
software/config/hardware, CPU affinity, threads and visible resource limits are
recorded per scope. Host RAM/CPU counts are visible capacity, not an allocation.
Unavailable cgroup v2 limits are recorded as null rather than invented quotas.

The opt-in tuner has explicit trial/time/step limits, seeded isolated model and
optimizer state, FP32 output/score/loss/gradient checks, and two real learning
updates before timing. It considers worker persistence/prefetch, supported BF16,
microbatch/accumulation tradeoffs at fixed effective batch, CUDA pinning/fused
optimization when available, and compilation. Candidate count and actual tested
candidates are recorded; a short run does not test the full search space.
The correctness probe forwards and backpropagates each microbatch separately,
weights losses by actual sample counts, and joins only detached outputs on CPU.
Its device forward batch never grows to the accumulated effective batch. As with
training, mixed label validity or dropout can make different batch partitions
numerically different; such a candidate must still pass the comparison.

Every timed candidate has two measurements. To replace the incumbent, its slower
repeat must beat the incumbent's faster repeat by more than 3%, with at least 10%
GPU memory headroom if applicable. An unsupported precision or failed numerical
probe is recorded as rejected. OOM/runtime failures release candidate state and
CUDA cache before continuing. The selected claim is always **best measured among
tested candidates**, with the baseline retained if a gain is unproven.

The time budget is a soft deadline checked between candidates and timed steps.
Worker loading is additionally bounded by `loader_timeout` (30 seconds by default,
capped to the remaining candidate budget in tuning). Kernel initialization,
probes, compilation, worker teardown and one atomic optimizer update can overrun
that deadline. Compilation is opt-in through a sufficiently large trial limit;
the bounded evidence run did not compile. This is not a hard subprocess watchdog.
Recheck throughput on the intended data/hardware before accepting a change.
