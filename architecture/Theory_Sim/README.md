# theory_sim — Incremental Simulation Program

This folder contains the incremental simulation program used in the development of GADS_Theory. The simulators isolate each subsystem's contribution by activating one additional feedback pathway at a time, producing dynamics that can be observed without entanglement from the other mechanisms that would be active in the full running system.

The program serves two roles. It tests predictions made in advance — some from "What Kind of Thing is MEGA," some from the formal flow-mapping that became GADS_Theory. It also surfaces dynamics that were not predicted, which are documented in Section 8 of GADS_Theory.

---

## The Core Argument: Layers 1, 5, and 6

The program contains seven layers (0 through 6), each isolating one additional mechanism. The three that carry the central theoretical result are Layers 1, 5, and 6. Read as a sequence, they demonstrate something specific: that differential survival is constitutive of representational evolution, and that the minimum possible fitness signal is sufficient to bootstrap the full architecture into self-regulation.

### Layer 1: The Base Energetic Engine

Layer 1 activates capture on top of the Layer 0 boundary mechanism. No Open operator, no metabolic pipeline, no fitness signal. This is the base energetic engine running unchecked — compression is present, but nothing releases the pressure it builds.

The result is the Grammar-Density Singularity. Within a few hundred generations, the population collapses into clones of a single hyper-compressed token. Recombination halts because depth-0 cut points vanish. Physical length approaches 1 while grammatical depth diverges. The system has done what it does, maximally, and it has done so catastrophically.

This establishes what the engine *is* by showing what it does in the absence of any counter-force. Capture produces grammatical depth at the cost of physical extent; without a dissipation channel, that trade-off runs until there is no extent left.

### Layer 5: The Pipeline Without Direction

Layer 5 activates every dissipation mechanism the architecture contains: Open, MCO temporal decay, the Participation Register, the Deletion Basket with inlining, and EA Recycling. The full metabolic pipeline is present. But fitness is not.

Without fitness, the pipeline runs — captures occur, deletions occur, compositions cycle through the register and basket — but its activity has no direction. The pipeline can remove pressure, but it has no signal telling it which pressure should be removed and which should be preserved. Participation in Layer 5 is just "appears in the population" with no filter on which organisms matter.

The system operates extensively. It does not collapse the way Layer 1 does. But eventually it halts under sustained pressure — the dissipation mechanisms alone do not produce the asymmetry that turns activity into direction.

This is the load-bearing observation: a fully-specified metabolic pipeline, complete with every mechanism the architecture contains for releasing compressive pressure, is not enough. Something more is required.

### Layer 6: Minimum Fitness Bootstraps the Architecture

Layer 6 adds the smallest possible fitness signal: decoded length. No semantic content, no task, no goal beyond "decoded length within this range." The fitness function is F(x) = min(L, T) − 2 · max(0, L − T), where L is decoded length and T is a target threshold. One point per unit up to threshold, minus two per unit over. No interpretation of content. No reward for solving anything.

This is not a test of whether MEGA can solve a task. It is a test of whether the minimum possible differential-survival signal is sufficient to give the dissipation pipeline direction.

It is. The same pipeline that was aimless in Layer 5 becomes directed in Layer 6. Compositions that don't contribute to surviving organisms are preferentially removed. Useful structure forms without being specified — the pipeline's own activity, now filtered by differential survival, produces representational organization. The system enters sustained oscillatory dynamics: two regimes, continuous transitions between them, no collapse, across thousands of generations.

The fitness signal is doing something specific and minimal. It is providing the smallest asymmetry that can distinguish organisms from each other. That asymmetry propagates through the participation mechanism into the pipeline, and the pipeline converts its own activity into direction. The fitness signal does not supply the structure. It supplies the gradient that the architecture uses to generate structure.

Size was chosen deliberately. Any semantic fitness function would obscure this result by adding task-specific optimization on top of architectural behavior. Size is the smallest gradient that distinguishes organisms, and the result holds at that minimum. This is evidence that self-regulation in GADS is not an emergent property of the task — it is a property of the architecture that requires only a minimal differential-survival gradient to activate.

---

## All Seven Layers

| Layer | Mechanism added                            | What it isolates                                                  | GADS_Theory §       |
|-------|--------------------------------------------|-------------------------------------------------------------------|---------------------|
| 0     | Boundary geometry alone (no capture)       | Phase regimes of the boundary insertion/removal regulator         | §8.1                |
| 1     | Capture                                    | Unconstrained compression → Grammar-Density Singularity           | §8.2                |
| 2     | Uniform compositional injection            | Path-dependent failure under global probability control           | §8.3                |
| 3     | Open operator                              | Outward force counteracting capture-driven compression            | forthcoming         |
| 4     | MCO (exponential temporal decay)           | Recency-biased selection binding compositions to current context  | forthcoming         |
| 5     | Full metabolic pipeline (PR + DB + inline) | Capacity-bounded persistence without direction                    | forthcoming         |
| 6     | Fitness function and differential survival | Closing the full GADS feedback loop                               | forthcoming         |

Layers 3–6 are implemented and produce reproducible output; their formal analysis in GADS_Theory Section 8 is in progress.

---

## Repository Contents

```
theory_sim0.py              Layer 0: boundary geometry alone
theory_sim1.py              Layer 1: capture added
theory_sim2.py              Layer 2: uniform compositional injection
theory_sim3.py              Layer 3: Open operator
theory_sim4.py              Layer 4: MCO temporal decay
theory_sim5.py              Layer 5: full metabolic pipeline
theory_sim6.py              Layer 6: fitness and differential survival

theory_sim_L1_delta_wav.py  Layer 1 variant with delta-waveform export
theory_sim_L6_delta_wav.py  Layer 6 variant with delta-waveform export

render_sound.py             Renders Layer 1 boundary dynamics as audio
render_sound_L6.py          Renders Layer 6 boundary + fitness dynamics as audio

analyze_sweep_sim_2.py      Tabulates results across Layer 2 parameter sweeps
```

Each layer simulator is standalone. Layers 1 and 6 have companion `_delta_wav.py` variants that add per-generation delta export for sonification; the plain versions are for analytical plots only. The render scripts consume the delta CSVs produced by the `_delta_wav` variants and produce stereo WAV files.

---

## Running the Simulators

```
python theory_sim0.py
python theory_sim1.py
...
python theory_sim6.py
```

Default parameters are set in the constants block at the top of each script. Key parameters:

- `POPULATION_SIZE`, `GENERATIONS`, `SEED` — basic run configuration
- `BOUNDARY_INSERT_PROB`, `BOUNDARY_REMOVE_PROB` — Layer 0 regulator
- `CAPTURE_PROB`, `MIN_CAPTURE_LEN` — Layer 1 compression
- `COMP_CHOICE_PROB` — Layer 2 ρ parameter
- `OPEN_PROB` — Layer 3 outward force
- `MCO_DECAY` — Layer 4 temporal bias
- `PR_CAPACITY`, `DB_THRESHOLD`, `MAX_INLINE_LEN` — Layer 5 metabolic pipeline
- `ELITE_FRAC`, `FITNESS_THRESHOLD`, `OVER_PENALTY` — Layer 6 selection
- `DYNAMICS_START_GEN` — generation at which delta-waveform plots and audio export begin (excludes the structural bootstrapping phase)

Parameters are accessible at the top of each file so that a reader exploring the dynamics can vary them and see the effects directly.

---

## Generating Audio

The `_delta_wav` variants export per-generation first-difference CSVs to a `sound_data/` subfolder. The render scripts turn those CSVs into stereo WAV files using FM synthesis — each dynamical signal drives one carrier voice, normalized to its own peak so that relative magnitudes are preserved without any single signal dominating the mix.

```
python theory_sim_L1_delta_wav.py && python render_sound.py
python theory_sim_L6_delta_wav.py && python render_sound_L6.py
```

Layer 1 renders four voices (contact rate, expansion-ready rate, free space, contact flux) to a single stereo file. Layer 6 renders eight voices organized into two groups — boundary dynamics (same four as Layer 1, for continuity) and fitness/selection dynamics (mean fitness, mean decoded length, fraction over threshold, service set size) — and produces three files: boundary only, fitness only, combined.

Mean gap is excluded from audio rendering despite being present in the CSVs. It is a different physical quantity (spatial extent) from the rendered signals (state-transition rates), and mixing them into the same stream produces a misleading composite.

---

## Interpreting the Output

Each analytical simulator produces:

**Standard plots:** token fractions, boundary geometry (A₀/A₁/A₂), segment length statistics, contact dynamics (f₀, f₁, Φ), DAG structure (nodes, edges, depth, connected components), and — at later layers — metabolic activity, MCO selection dynamics, and fitness distributions. These answer "what happened during this run."

**Delta-waveform plots (on `_delta_wav` variants):** per-generation first differences of the primary dynamical signals. Raw signal values are dominated by their static level; deltas strip the baseline and leave only the events. This is where the qualitative character of each layer's dynamics becomes visible. A layer that has collapsed into singularity shows flat deltas. A layer in sustained non-equilibrium shows continuous small-scale deltas punctuated by larger events.

The audio output makes the same information audible. A system at the grammar-density singularity sounds silent. A system in the Layer 0 negotiation regime sounds like periodic activity against a quiet background. A system with the full feedback loop closed sounds like continuous bubbling activity — structured but non-repeating. Sonification is a diagnostic tool; some patterns are easier to hear than to see.

---

## Layer 2 Sweep Analysis

`analyze_sweep_sim_2.py` processes the output folders produced by multiple Layer 2 runs at different `COMP_CHOICE_PROB` (ρ) values and tabulates:

- Singularity generation (when composition fraction first crosses 0.5)
- Maximum MG size reached
- A₀ variance
- Boom-bust cycle count in A₀

This is the methodology behind Proposition 8.3 of GADS_Theory: the claim that no static ρ reliably bounds both physical length collapse and grammatical divergence across stochastic paths. The sweep demonstrates that behavior is not ρ-determined but path-dependent. Running Layer 2 at several ρ values with different seeds and then running the analysis script reproduces the table supporting the proposition.

---

## What to Look At First

If you're engaging with this material for the first time and want the core result:

1. Run `theory_sim1.py`. Observe the Grammar-Density Singularity — composition fraction crosses 0.5, dominant-composition fraction approaches 1.0, recombination halts as depth-0 cut points vanish. This is the base engine running without dissipation.

2. Run `theory_sim5.py`. Observe the full metabolic pipeline operating — captures, deletions, rescues cycling through the register and basket — but eventually halting under sustained pressure because nothing gives the pipeline direction.

3. Run `theory_sim6.py`. Observe sustained oscillatory dynamics across 5000 generations. The same pipeline that was aimless in Layer 5 is now directed by the minimum possible fitness signal.

The three runs together are the argument: the engine alone collapses, the pipeline without fitness halts, the pipeline with minimum fitness self-regulates. The other four layers (0, 2, 3, 4) show why each intermediate mechanism is needed and what specific failure mode each addresses.

For the audio version, run `theory_sim_L1_delta_wav.py` followed by `render_sound.py`, then `theory_sim_L6_delta_wav.py` followed by `render_sound_L6.py`. Listening to Layer 1 end in silence and Layer 6 sustain continuous activity is the same argument in a different sense.

---

## Relationship to GADS_Theory

The simulators and GADS_Theory co-evolved. Predictions made in the blog post and the theoretical flow-mapping were tested here. Dynamics that the simulators surfaced without prediction were documented in Section 8. The Layer 2 section of Section 8 is the clearest example of the interplay: a prediction was stated in advance (ρ as a linear modulator of the singularity timeline), the simulator falsified it, the failure was diagnosed (temporal unbinding of compositional structure from evolutionary context), and the diagnosis produced a new hypothesis that Layer 4 subsequently tested.

Section 8 currently presents Layers 0–2 in full formal detail. Layers 3–6 are implemented in the simulator and produce reproducible output; their formal analysis is being written up as the current phase of the work.

---

## Dependencies

- Python 3.8+
- `numpy`
- `matplotlib`
- `scipy` (only for the render scripts, which use `scipy.io.wavfile` and `scipy.interpolate`)

No external data, no API calls, no GPU. A full Layer 6 run (5000 generations, population 500) completes on a modest CPU in a few minutes.