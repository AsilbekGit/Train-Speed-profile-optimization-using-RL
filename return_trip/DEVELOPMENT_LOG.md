# Return-Trip PPO — Development Log & Known Issues

Purpose: a running record of how the **return-trip (B → A) PPO** experiment was
built, what went wrong along the way, and what was changed to fix it. Written so
the issues, dead-ends, and design decisions can be cited honestly in the paper —
including the runs that did **not** complete the route.

Scope note: this file is intentionally **PPO-only**. DQN / Q-SARSA are not
discussed or compared here.

---

## 1. What the return trip is

The forward route (`data/data.csv`) runs A → B over 749 segments × 100 m = 74.9 km.
The return trip traverses the **same physical track in reverse** (B → A):

- segment order reversed,
- **grade signs flipped** (a forward +x% uphill becomes a −x% downhill on return),
- speed limits and curvature reversed **without** sign change (direction-agnostic).

Implemented in `return_trip/route.py` (`load_reverse_route()`), fed through the
existing `ppo/env_wrapper.py:PPOTrainEnv`. The algorithm/network code is identical
to the forward PPO (`ppo/train.py`); only the route data differs.

**Reversal correctness was verified** (not assumed):
- `reverse_stations == mirror(forward_stations)` → True (exact index mirror).
- `reverse_grade[0] == -forward_grade[-1]` → True (sign flip + reversal).
- Grade range mirrors: forward `[-4.63%, 6.28%]` → reverse `[-6.28%, 4.63%]`.

So the route transformation itself is **not** a source of error.

---

## 2. Result timeline (PPO)

| Run | Config | Steps | Result | Energy | Notes |
|---|---|---|---|---|---|
| Forward (A→B) | default (`limit_term=5.0`, `energy_coef=2.0`) | 1.5M | **748/749 COMPLETE** | 679 kWh | reference run, `ppo/results/` |
| Return v0 (baseline) | default | 1.5M | **718/749 INCOMPLETE** | ~110 kWh | stalls/terminates at seg 718 |
| Return v0b | `--energy-coef 1.0` | 1.5M | **718/749 INCOMPLETE** | 172 kWh | softer energy penalty changed nothing → energy was not the cause |
| Return v1 (quick fix) | `limit_term` DISABLED, `--limit-pen 10` | 1.5M | **KILLED @ epoch 17** (OOM) — not demonstrated | — | superseded; expert review flagged `limit_pen=10` as a fragile break-even regime |
| Return v2 (principled) | + `dist_to_tight_limit` obs feature, `--limit-pen 4`, station relief, termination off | 1.5M | **748/749 COMPLETE** ✅ | 99.4 kWh | the change documented in §5; details in §6 |

The incomplete baseline (Return v0) is preserved as failure-mode evidence at
`results_cm/return/ppo_baseline_incomplete/` (copied before the fixed re-run).

The Return v1 quick-fix run was **OOM-killed at epoch 17/75** — not a code fault;
it was running on the GB10's unified memory concurrently with two heavyweight
(Opus) code-review agents. Lesson: run training **solo**. It was superseded by the
principled fix (v2) after expert review, so it was never re-run on its own.

---

## 3. Root cause of the incomplete return trip

The return policy **converges confidently** (test-return curve plateaus flat at
~8000; it is not under-trained) but stops at **segment 718 / 749 (96% of the
route)**. Diagnosis:

1. The reverse route has a **3.0 m/s speed-limit zone at segment 718**, immediately
   before a station at 719 (1.0 m/s). This is the mirror of an *easy* zone on the
   forward trip (forward seg ~30), which the forward trip handles trivially because
   there it sits near the **start**, where the train is still slow.
2. On the return trip that same tight zone sits near the **end**, reached at
   **~18 m/s** after a long open-limit cruise (segs ~684–717 allow 19.4 m/s).
3. Physics: `MAX_DEC = -0.8 m/s²` (`env_settings/config.py`). Braking 18 → 3 m/s
   needs ~197 m ≈ **2 segments** of full braking. The policy does not anticipate
   it (cruises at 17.9 m/s into seg 718).
4. `ppo/env_wrapper.py` **hard-terminated the episode** when the train exceeded a
   limit by more than `limit_overshoot_term = 5.0 m/s` (overshoot here ≈ 15 m/s).
   The episode was killed at seg 718, so **completion was unreachable**.
5. Why the policy never learned to brake: it was a stable **local optimum**.
   Precise framing (corrected after expert review — *do not* state it as the loose
   "−50 vs 8000" ratio): reaching seg 718 already banks ≈ **9080** of base reward
   (progress `+0.1·ds`, `+2`/segment, `+0.1` per step at `v>5`). *Completing* the
   route adds only the **+1000 completion bonus** plus ≈ +390 for the last 31
   segments (≈ +1390 gross), and to collect it the policy must brake hard, crawl a
   station, and re-accelerate — sacrificing progress/speed reward and risking the
   station stall-trap (§5). The flat −50 termination added almost nothing either
   way. So the honest statement is: **the +1000 completion bonus was too small
   relative to the ~9080 already earned at 96% to justify the narrow early-braking
   maneuver.** This is also why lowering `--energy-coef` (Return v0b) did nothing —
   it was never an energy-reward problem.

In one line: **a single tight-limit zone near the end of the reversed route,
combined with a hard episode-termination on overshoot, made it strictly better
for the converged policy to crash the limit and stop at 96% than to learn the
narrow early-braking maneuver.**

---

## 4. Process mistakes made during development (for honest disclosure)

- **Lost the original return results once.** A quick `--steps 8000` sanity run was
  accidentally pointed at the real output folder and overwrote the first full
  return run. Because training is seeded (`seed=0`), the full result was
  **regenerated deterministically**, so no scientific information was lost — but
  it is recorded here as a process error. (Output folders are now copied aside
  before destructive re-runs.)
- **`training_curves.png` was blank on early runs — a plotting bug, not a training
  failure.** `ppo/plot.py:plot_training_curves` looked up logger keys with
  `test/` / `train/` / `update/` prefixes (e.g. `test/returns_stat/mean`,
  `update/loss`), but the in-memory logger stores **bare** keys
  (`returns_stat/mean`, `loss/mean`) — the prefix only selects the bucket, it is
  not prepended to column names. Every lookup missed → empty axes. A second issue
  surfaced when regenerating from CSV: blank cells (steps with no finished episode)
  crashed `float('')`. Both fixed in `ppo/plot.py`; curves regenerated from the
  existing CSVs without retraining. Any training run **before** this fix has a
  meaningless `training_curves.png` even though the underlying CSV data is valid.
- **`ppo_v2` naming / stray results folder.** Early on there were two PPO variants
  (a 2-D-observation baseline and the 8-D version kept here). The 2-D baseline and
  the "v2" suffix were removed; PPO results now live in `ppo/results/` (forward)
  and `results_cm/return/ppo/` (return). Mentioned only so old commits referencing
  `ppo_v2` are not mistaken for a third model.

---

## 5. The fix applied — PRINCIPLED (after expert review)

Two Opus-class expert agents reviewed the first quick fix (disable termination +
`limit_pen=10`). Verdict: the env change was *correct and backward-compatible*,
but the quick fix was a **band-aid**. Key points adopted here:

- A route that "completes" by driving **18 m/s through a 3 m/s zone** is, for a
  *speed-profile* paper, **worse than an honest incomplete trip** — respecting the
  speed limit is a hard constraint of the problem, not a soft objective.
- `limit_pen=10` sits at a **break-even**: plowing both 3.0 zones costs ≈ −1400,
  finishing gains ≈ +1390 → fragile / seed-dependent, and it can convert the
  "plow-through" failure into a new **stall-at-station** failure (the base env
  charges −10/step for `v<1` at a station, and the wrapper truncates after 200
  stalled steps).
- The agent was **timing-blind**: the observation gave the *tightest limit within
  20 segments* but not **how far away** it is, so it could not decide whether to
  brake now or in fifteen segments — capping how well any penalty-only fix can do.

**Code changes (this is what is now in the repo)**

1. `ppo/env_wrapper.py` — **new 9th observation feature** `dist_to_tight_limit`
   (feature index 7): normalized segments to the nearest tightest limit in the
   20-segment window. This is the timing signal that was missing. `OBS_DIM` 8 → 9.
   *(Changes the network input, so BOTH forward and return are retrained.)*
2. `ppo/env_wrapper.py` — **station stall-trap relief**: when the train is
   legitimately crawling (`v<1`) through a station segment, add `+10` to cancel the
   base env's −10/step, so braking *for* a station no longer bleeds reward (which
   was an incentive to blow through at speed). Applied in the wrapper only;
   `env_settings/` is untouched.
3. `ppo/env_wrapper.py` — hard overshoot termination is **optional**
   (`limit_overshoot_term=None` disables it). Default stays `5.0`, so the forward
   trip's *execution path* is unchanged.
4. `return_trip/train_ppo.py` — return defaults: termination **disabled**
   (`--limit-term -1` → `None`) and graded penalty **`--limit-pen 4`** (down from
   the fragile 10; up from forward's 2). With the new timing feature, 4 is enough
   pressure to brake without making stall-at-station attractive.

**Rationale.** Removing the cliff makes completion *reachable*; the timing feature
makes limit-compliant completion *learnable*; the moderate graded penalty + station
relief keep the incentives away from both degenerate corners (plow-through and
stall-at-station).

**Alternatives considered (and why not chosen)**

- *Keep termination, raise penalty to ≈ −1500.* Still a non-differentiable cliff
  the policy already learned to ignore — "the thing that failed, scaled up."
- *Quick fix only (no obs change).* Cheaper (no forward retrain) but leaves the
  agent timing-blind; expert review judged it likely to swap one degenerate policy
  for another.

**Honest limitation that remains.** Even with the timing feature, limit compliance
at seg 718 is driven by a *penalty*, not a hard constraint — so the paper should
report the actual over-limit step count at that zone (now logged in `speed_log.csv`
as `over_limit_ms`, and marked red on both plots) rather than claim guaranteed
compliance.

---

## 6. Results — IDENTICAL model, both legs (9-D obs, limit_pen 4, termination off, 1.5M steps)

These are the final matched results: forward and return use the same `PPOTrainEnv`
and network (§7d); only the route differs.

| Metric | Forward (A→B) | Return (B→A) |
|---|---|---|
| Final segment | **748/749 COMPLETE** | **748/749 COMPLETE** ✅ (was 718) |
| Energy | 682.6 kWh | 99.4 kWh |
| Time | 5881 s | 5173 s |
| Actions | Brake 0.6 / Coast 42.3 / Cruise 49.2 / Power 7.9 % | Brake 4.6 / Coast 55.6 / Cruise 38.4 / Power 1.3 % |
| Over-limit steps (total) | 43 | 460 |
| Behavior at seg 714–722 (old failure zone) | n/a (forward passes it slow, near start) | slows to **0.6 m/s** (brakes for the station); 23 residual over-limit steps on the approach |

**The return trip now completes the route** — the timing feature lets it brake for
the seg-718 zone instead of being killed there (it slows to ~0.6 m/s vs the old
17.9 m/s blow-through). Compliance is much improved but **not perfect**: the return
has 460 over-limit steps vs the forward's 43 (its tight zone is approached fast and
compliance is penalty-driven, not a hard constraint). This is the honest result —
directly auditable in `speed_log.csv` (`over_limit_ms`) and shown as red markers on
both PNGs.

Note on energy: the 682.6 vs 99.4 kWh gap is now a legitimate, **route-only**
comparison (same model — §7d) and is physically expected — the return is net
downhill so it coasts far more (55.6% vs 42.3%) and powers far less (1.3% vs 7.9%).
See §7c for the full elevation accounting.

**The return trip now completes the route** — the timing feature lets it brake for
the seg-718 zone instead of being killed there (it slows to ~0.6 m/s vs the old
17.9 m/s blow-through). Compliance is much improved but **not perfect**: ~23
over-limit steps remain on the *approach* to that zone (max 7.1 m/s over), because
compliance is penalty-driven, not a hard constraint. This is the honest result to
report — and it is now directly auditable in `speed_log.csv` (`over_limit_ms`
column) and visible as red markers on both PNGs.

Note on energy: now that **both legs complete**, the 99 vs 677 kWh gap is a
legitimate comparison and is physically expected — the return leg is net *downhill*
(grades sign-flipped) so it coasts far more (55.6% vs 38.6%) and powers far less
(1.3% vs 8.6%). Still report it with the over-limit caveat above.

Artifacts: `results_cm/return/ppo/{speed_profile.png, speed_vs_time.png,
speed_log.csv, training_curves.png}`; forward in `ppo/results/`; the 8-D forward
baseline preserved at `ppo/results_obs8_baseline/`.

---

## 7. Reproducibility

```bash
# Forward reference (unchanged behavior, termination at 5 m/s):
python -m ppo.train --steps 1500000                 # -> ppo/results/

# Return trip, fixed (termination disabled, strong graded limit penalty):
python -m return_trip.train_ppo --steps 1500000     # -> results_cm/return/ppo/

# Reproduce the INCOMPLETE baseline for the paper's failure-mode figure:
python -m return_trip.train_ppo --steps 1500000 --limit-term 5 --limit-pen 2
```

All runs use `seed=0` and require a CUDA GPU.

---

## 7b. New visualization & logging outputs (added this round)

To make the train's behavior over the route easier to read and report, every PPO
run (forward and return) now writes, via the shared `ppo.plot.save_rollout_outputs`:

- **`speed_log.csv`** — one row per env step (= 1 s): `t_s, segment, position_km,
  speed_ms, speed_kmh, speed_limit_ms, over_limit_ms, action, action_name,
  energy_kwh`. The train's current speed at every second is now directly
  inspectable, and `over_limit_ms` quantifies any speed-limit violation.
- **`speed_vs_time.png`** — speed + active speed limit + action strip vs **time**
  (the existing `speed_profile.png` is vs **distance**). Makes accel/brake phases
  and station dwell obvious on the clock.
- **`speed_profile.png` enhancements** — tight non-station low-limit zones (≤5 m/s)
  are shaded, and any step where the train exceeds its segment's limit is marked
  in red (with a count in the legend). The seg-~718 failure zone is now visually
  obvious.

These are presentation/diagnostic only — they do not affect training.

## 7c. Energy asymmetry (676 vs 99 kWh) is physics, not a bug

A reasonable question: why does the return use **6.8× less energy** than the
forward? Checked against the route data — it is correct:

- The route is strongly **net uphill A → B**: `sum(grade·dx) = +346.8 m` (444 m of
  total climb vs only 97 m of descent over 74.9 km).
- So **B → A is net downhill by 347 m**. The pure gravitational energy of that
  elevation difference is `m·g·h = 360 t · 9.81 · 347 m ≈ 340 kWh`.
- The energy model has **no regenerative braking** (`environment.py:91` charges
  energy only when `f_trac > 0`; on a downhill the grade resistance
  `m·g·(grade/100)` is *negative*, so `Cruise` computes `max(0, …) = 0` traction).
  The descent energy is therefore neither spent nor recovered — the train coasts.

Result: the forward leg must power up 444 m of grade (≈ 510 kWh of gravity work +
resistance → 676.7 kWh); the return leg only powers the ~97 m of uphill bumps and
coasts the rest (→ 99.4 kWh). **The ~340 kWh net-elevation difference plus no regen
fully explains the gap.** It is a property of the route + model, not a code error.
(If a future version models regenerative braking, the return number would rise.)

## 7d. Same model for both legs (forward == return except the route)

To make the two legs directly comparable, the forward (`ppo/train.py`) and return
(`return_trip/train_ppo.py`) trainers now build an **identical** `PPOTrainEnv` and
network — the *only* difference is the route data:

| | value (both legs) |
|---|---|
| Observations | 9-D (incl. `dist_to_tight_limit`) |
| Network | MLP [256, 256], tanh |
| `energy_coef` | 2.0 |
| `limit_pen_coef` | 4.0 |
| limit-overshoot termination | DISABLED |
| station stall-relief | on |

Previously the forward kept hard termination (5 m/s) and `limit_pen=2`, which
confounded the comparison. The forward was retrained under the unified config; the
return already used it. With the same model, the remaining 676 vs ~99 kWh gap is
attributable purely to the route (§7c), which is the point.

## 7e. "Less aggressive" reward — tried, then REVERTED as too harsh

A gentler-driving reward was tried (energy_coef 2→4, cancel the `+0.1` v>5 speed
bonus, and a `-0.1·|Δaction|` jerk penalty for smoothness, all applied identically to
both legs). Stacking all three was **too harsh** — it over-suppressed powering and
risked the route not completing — so it was reverted.

**Final reward = the simple one (route completes reliably):**
`energy_coef=2`, `limit_pen=4`, termination disabled, station relief, 9-D obs, and
**no** jerk penalty / **no** speed-bonus cancel. Both legs complete (§6).

The optional knobs remain in `PPOTrainEnv` for future, careful tuning but are **off by
default**: `jerk_pen_coef=0.0` (also `--jerk-pen`, default 0) and
`cancel_speed_bonus=False`. The base env (`env_settings/`) is untouched, so any future
use of these does not affect DQN/Q-SARSA. Lesson for the paper: gentleness vs.
guaranteed completion is a real trade-off here; with a completion-dominated reward,
piling on penalties can flip a completing policy into a non-completing one, so
gentleness should be added one mild knob at a time and verified.

## 7f. Reward simplified further — dropped the station-relief term

By request, the reward was trimmed once more. The `+10` station-relief term (a
special-case bonus that cancelled the base env's −10/step "stopped" penalty at
stations) was removed. With the `dist_to_tight_limit` feature the train brakes for a
station in time and just **creeps through at ~1–3 m/s** (above the base −10's `v<1`
band and below the base −5's `v>limit+2` band), so it no longer needs a stop-and-sit
bonus.

**Final reward (simplest so far):**
- Base env (`environment.py`, shared, untouched): progress `+0.1/m`, `+2`/segment,
  `+0.1` if v>5, `−5` if over limit+2, `−10` if stopped, `+1000` completion.
- PPO wrapper adds exactly two terms: **energy penalty** `−2·energy_step` and a
  **proportional limit-overshoot penalty** `−4·overshoot`.

That's it — no station relief, no jerk penalty, no speed-bonus cancel (the last two
remain as off-by-default optional knobs). The pre-trim runs (with station relief) are
kept as a fallback at `/tmp/ppo_fallback/` until the simpler reward is confirmed to
still complete both legs; results filled into §6 after the retrain.

## 8. Open items to mention in the paper

- Reward design is **completion/progress-dominated**; the energy term is small by
  comparison, so PPO optimizes primarily for finishing and speed, with energy as a
  secondary nudge (`env_settings/environment.py` base reward + `energy_coef`).
- Start speed is **hardcoded at 15 m/s** (`environment.py:reset`); the agent never
  departs from rest. Relevant when interpreting behavior at the first/last segments.
- **No regenerative braking** is modeled — braking and coasting are zero-energy;
  kinetic energy spent under Power is not recoverable. This is why efficient
  policies favor coasting over brake-then-repower.
- The reverse route is *harder to learn* than the forward route purely because of
  **where** its tight-limit zone falls (near the end, approached fast), not because
  of any asymmetry in the physics or an error in the reversal.
- **Do not report the forward-vs-return energy comparison (e.g. 679 vs 172 kWh)
  until both legs complete the route.** The incomplete return numbers come from a
  run that quits *before* the costly final station approach, so the comparison is
  not apples-to-apples.
- **`success`-threshold inconsistency:** `return_trip/train_ppo.py` / `ppo/train.py`
  print "COMPLETE" at `final_seg >= n_segments-2` (747), while the env's
  `info['success']` uses `>= n_segments-1` (748). Pick one definition and report it.
- The two trainers share a duplicated `_NormalizedSingleEnv` (eval-time obs
  normalization); the rollout *output* code is now de-duplicated via
  `save_rollout_outputs`, but the normalization helper is still copied in both.
