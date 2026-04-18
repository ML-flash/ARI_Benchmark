# ARI Benchmark

**Attractor Resolution Index Benchmark for Measuring Constructive Agency**

Submitted to the [Measuring Progress Toward AGI — Cognitive Abilities](https://kaggle.com/competitions/kaggle-measuring-agi) hackathon (Executive Functions track), hosted by Google DeepMind and Kaggle.

This repository contains the MEGA (Mutable Encoding Genetic Algorithm) implementation used as the non-LLM baseline in the benchmark. The Kaggle Benchmark task itself (which evaluates LLMs) is hosted on the Kaggle Benchmarks platform. This repo provides the environment code, the ARI analysis pipeline, and the evolutionary baseline that brackets the performance range reported in the writeup.

---

## What This Benchmark Measures

The ARI Benchmark tests whether an agent can **systematically restructure a black-box environment** in ways that improve its own future outcomes — a capacity we call *constructive agency*. This is distinct from planning within a fixed problem space: the agent's actions change the problem itself.

An agent navigates a 3D toroidal grid containing items with chemical interactions, collects items subject to knapsack constraints, and can **drop items to reshape the spatial layout** for future traversals. The benchmark measures whether this restructuring is *directed* — whether entropy reduction in the environment correlates with fitness improvement — using the **Attractor Resolution Index (ARI)**:

```
ARI = ρ(rolling_entropy_reduction, rolling_best_fitness) × (max_cumulative_reduction / H₀)
```

ARI is bounded to [-1, 1]. Positive values indicate constructive agency; negative values indicate destructive restructuring; near-zero indicates no coherent environmental modification.

---

## Repository Structure

```
├── TSS_benchmark.py            # Small-scale environment (volume=15, 200 items)
├── TSS_benchmark_runner.py     # Runner + ARI analysis for small scale
├── TSS_Benchmark_Large.py      # Large-scale environment (volume=45, 2000 items)
├── TSS_Runner_Large.py         # Runner + ARI analysis for large scale
├── src/
│   └── M_E_GA/                 # MEGA framework (Mutable Encoding Genetic Algorithm)
│       ├── M_E_GA_Base.py      # Core GA orchestrator
│       ├── M_E_Engine.py       # Encoding manager (gene/metagene operations)
│       ├── GA_Logger.py        # Event logging
│       └── engine/             # Modular engine components
│           ├── population_manager.py
│           ├── mutation_manager.py
│           ├── crossover_manager.py
│           ├── gene_manager.py
│           ├── meta_gene_manager.py
│           ├── organism_generator.py
│           ├── logging_manager.py
│           └── mutation/
│               ├── basic_mutations.py
│               ├── delimiter_mutations.py
│               └── metagene_mutations.py
├── setup.py
└── LICENSE                     # GPLv3
```

---

## Environment Details

| Parameter | Small Scale | Large Scale |
|-----------|------------|-------------|
| Grid volume | ±15 (31³ = 29,791 positions) | ±45 (91³ = 753,571 positions) |
| Items | 200 | 2,000 |
| Groups | 5 | 5 |
| Max size / weight / density | 80 / 100 / 25 | 80 / 100 / 25 |
| Soft step limit | 50 | 200 |
| Actions | R, L, U, D, F, B, DR (drop) | R, L, U, D, F, B, DR (drop) |
| Environment seed | 42069 | 42069 |

The small-scale configuration matches the Kaggle Benchmark task (70 generations, 100 paths per generation). The large-scale configuration is provided for extended validation runs.

### State Space: No Global Gradient, Rugged Local Gradient, Self-Generated Problems

The design property that makes this benchmark cognitively demanding is the interaction of three features of the state space: there is no globally exploitable gradient, the local gradient within any fixed instance is rugged and trap-laden, and the only mechanism for reshaping the environment carries an immediate fitness cost that works against the agent's short-term interest. These three features are not independent — they compose. Understanding how they compose is essential to understanding what ARI actually measures.

#### No global gradient

The toroidal 3D grid has no privileged direction. Every position is topologically equivalent to every other. There is no origin, no boundary, no "center of mass" that a policy can orient around. The agent starts at `(0, 0, 0)` but this is an arbitrary labeling — wrapping coordinates makes the origin indistinguishable from any other cell.

Over the space of all possible configurations, all paths are equivalent in expectation. Consider two arbitrary action sequences P₁ and P₂, and consider the full ensemble of possible item placements (all ways of distributing items with their random properties and interactions across the grid). Averaged over this ensemble:

```
E[fitness(P₁) | config ~ Uniform] = E[fitness(P₂) | config ~ Uniform]
```

provided P₁ and P₂ have the same count of movement actions and the same count of drop actions. (Drop actions do not count as steps and do not move the agent — they are evaluated in a separate branch of the path loop, so two paths with identical movement-action counts incur identical step-reward budgets regardless of how many drops they contain.)

The toroidal symmetry and uniform item placement mean that whatever P₁ encounters in one instance, some permutation of items ensures P₂ encounters equivalent structure in another. Drops contribute an expected value equal to the drop reward times the probability of an unoccupied target cell, which is position-invariant under uniform item placement. No path has a prior advantage over any other path. A policy that tries to pattern-match against typical spatial regularities — "go right, that direction usually has more valuable items" — has no purchase here, because averaged across all instances, right is the same as left is the same as any diagonal.

This is a claim about the *class* of instances, not about how any single instance behaves. It means an agent cannot bring useful priors about where items cluster, which directions lead to richer regions, or what long sequences of actions are generally productive. The environment class offers no such regularities to learn in advance.

#### Rugged local gradient

Once `ENV_SEED=42069` instantiates a specific configuration, that configuration does contain exploitable structure. But the structure is rugged. The local gradient is not a smooth slope the agent can ascend — it is a jagged landscape of narrow peaks surrounded by cliffs, and the cliffs are where naive optimization lands.

- Some spatial regions contain denser item clusters than others, but traversing them efficiently requires sequences that are not obvious from local observation.
- Some small item subsets produce favorable chemistry cascades; adding or removing a single item from the subset can flip the cascade from favorable to ruinous.
- Constraint violations cascade: a single over-limit pickup triggers last-in-first-out removal with a 10-point penalty and full chemistry recalculation, which can change which items now violate constraints, which can trigger further removals.

Success in this landscape depends on the agent's ability to reconfigure the state space into one where its strategies work — not on the agent's ability to find an optimum in the initial configuration. The initial configuration is hard by construction. What the agent can do is make it less hard, but only if it does so coherently.

#### Drops cost the agent: the system creates its own problems while solving the current one

The drop action (DR) carries a trivial positive reward (0.001) per successful drop — the smallest reward signal in the environment by orders of magnitude. But the real economics of a drop are negative in immediate terms: dropping an item removes it from the agent's sack, which means the agent loses that item's value *and* reduces the exponential multiplier (1.75^n, where n is the number of retained items). A single drop can cost hundreds or thousands of fitness points on the current path. The 0.001 reward exists only to give a learnable gradient signal that drops are a legitimate action class — it does not come close to compensating for the immediate loss.

This means every drop is an immediate sacrifice. An agent that drops an item is accepting a concrete, measurable fitness penalty on the current path in exchange for a structural change to the environment whose payoff — if any — will only be realized through subsequent paths that traverse the reshaped space. The payoff is speculative, delayed, and contingent on the agent's future behavior being adapted to the new configuration.

This creates the benchmark's central tension. An agent that drops items is simultaneously:

1. **Degrading its current performance** — losing item value and exponential multiplier on the path that executes the drop.
2. **Dismantling the local topology it had adapted to** — the item distribution that made its current strategies productive is no longer the item distribution it faces.
3. **Creating a new problem for itself** — subsequent paths encounter a different environment, one that the agent's prior learning may or may not transfer to.

The third point is the key recursive property. **Each drop changes the problem the agent is solving.** An agent that drops without a coherent model of what the reshaped environment will look like is generating a sequence of novel problems for itself, each one partially invalidating what it learned from the last. The system creates its own new problems while solving the current one.

This is why constructive agency, rather than mere optimization, is what the benchmark isolates. The cognitive demands map directly onto the three sub-abilities of executive function defined in Burnell et al. (2025):

- **Planning** (Owen, 1997) — "formulating sequences of future actions to achieve specific goals." Each drop is an immediate fitness sacrifice. The dropped item's value and the exponential multiplier reduction (1.75^n) are lost on the current path. The only way a drop produces net value is if subsequent paths — which the agent has not yet generated — traverse the reshaped space in a way that recovers more than what was lost. The agent must plan across its own restructuring boundary: formulating future action sequences whose success depends on environmental state changes the agent itself is causing now.
- **Inhibition** (Bari & Robbins, 2013; Miyake et al., 2000) — "changing, withholding, or suppressing learned or habitual responses in favor of more controlled, goal-appropriate ones." The step reward (+2 per step for the first 50 steps) is the most reliable and immediately legible signal in the environment, producing ~100 fitness points with zero understanding of the deeper mechanics. An agent that has learned to associate longer paths with higher scores must suppress that habitual response to explore drops — actions that cost fitness now and whose benefit is neither immediate nor guaranteed. The step-reward plateau is the habitual response; constructive restructuring requires inhibiting it.
- **Cognitive flexibility** (Braem & Egner, 2018) — "the ability to switch between different tasks, concepts, or ways of thinking." Each successful drop changes the item distribution, which means the agent's task has changed. Strategies that worked before the drop may not work after it. The agent must detect when its own actions have invalidated its current approach and switch to one adapted to the new configuration — repeatedly, across a sequence of self-generated environmental changes. An agent that perseverates on a strategy fitted to a prior state of the environment will degrade its own performance.

The benchmark also draws on **learning** (acquiring new knowledge through experience): the agent receives only scalar fitness scores as feedback, observes them through a sliding 5-generation window, and must extract causal structure from this sparse signal to form and update hypotheses about an environment whose rules are never disclosed. But learning is instrumental here — it serves the executive functions above. The benchmark's primary signal, ARI, measures whether the agent's learned model of the environment is good enough to support *directed restructuring*, not whether the agent learned per se.

#### What ARI measures in this context

ARI does not measure whether an agent finds the optimum — the optimum is not well-defined, because the environment the agent ends with is not the environment it started with. ARI measures whether, over time, the agent's restructuring actions are *directed*: whether the entropy reduction it imposes on the environment correlates with improvements in what the agent can subsequently achieve.

- An agent that drops items randomly will reduce entropy (any clustering reduces entropy) without correlation to fitness → **near-zero ARI**.
- An agent that drops items strategically — accepting the immediate cost because it has a model of how the reshaped environment will improve future paths — will reduce entropy *and* improve fitness, in the same temporal pattern, generation over generation → **positive ARI**.
- An agent that drops frequently without a coherent model will reduce entropy while fitness degrades, because it is paying the immediate cost of drops while dismantling productive structure faster than it is creating it → **negative ARI**.

The Spearman rank correlation is the appropriate statistic because the relationship between entropy reduction and fitness improvement is monotonic but not linear — the agent is navigating a rugged landscape, not a smooth one, and rank correlation captures directional alignment without assuming proportionality. The multiplicative form (correlation × displacement) ensures that neither signal alone is sufficient: an agent must both restructure substantially *and* do so in alignment with fitness gains to register meaningful ARI.

### Fitness Computation

- Items are automatically picked up when the agent visits their position (no pickup action is required or available).
- At path termination, the sack is evaluated against three simultaneous knapsack constraints (size, weight, density).
- Items violating constraints are removed last-in-first-out, with a 10-point penalty per removal and full chemistry recalculation after each removal.
- Valid collections receive: `sum(item values) × 1.75^n` where `n` is the number of retained items. The exponential multiplier heavily rewards keeping more items, creating strong incentive to pack tightly.
- Steps earn +2 each for the first `soft_step_limit` steps, then -10 per step thereafter. This creates a deliberate step-reward plateau: an agent that discovers only the step reward and never learns the deeper mechanics can reliably score ~100 points from movement alone. The plateau distinguishes agents that collapse into low-risk local optima from agents that continue to improve.

### Chemical Interactions

Each item is assigned:
- A **group** (one of 5), which determines what reactions can target it.
- A **reaction strength** (sampled uniformly from `[0.01, 50.0]`), which scales how strongly its interactions affect others.
- A set of **interactions**, each specifying a target group, a property to modify (size, weight, density, or value), a direction (increase or decrease), and a magnitude.

When two items are co-located in the sack, interactions fire **bidirectionally** in pickup order — item A's interactions affect item B's properties, and item B's interactions affect item A's. Properties are clamped to a minimum of 0.1. The combined effect is that the value of any subset of items depends on the full interaction graph restricted to that subset, which makes the embedded packing problem computationally hard: small changes in composition can produce large, non-monotonic fitness swings depending on which chemistry cascades fire.

---

## Installation

```bash
git clone https://github.com/ML-flash/ARI_Benchmark.git
cd ARI_Benchmark
pip install -e .
```

### Dependencies

- Python 3.8+
- `xxhash`
- `numpy`
- `matplotlib`
- `scipy`

```bash
pip install xxhash numpy matplotlib scipy
```

---

## Running the Benchmark

### Small scale (matches Kaggle task parameters)

```bash
python TSS_benchmark_runner.py
```

Runs 70 generations with a population of 100. Produces:
- `ARI_analysis.png` — six-panel composite ARI analysis
- `fitness_over_time.png` — max fitness trajectory
- `panels/` — individual analysis panels (560×280 px)

### Large scale

```bash
python TSS_Runner_Large.py
```

Runs 2,000 generations with a population of 500. Same output structure.

### Configuration

Both runners expose GA parameters at the top of the file. Key settings:

- `GLOBAL_SEED` — set to `None` for stochastic runs, or an integer for reproducibility
- `ENV_SEED` — fixed at `42069` to match the Kaggle benchmark (do not change for comparable results)
- `max_generations`, `population_size`, `num_parents` — control run duration and population dynamics
- Mutation probabilities — tune evolutionary pressure

---

## Understanding the Output

### ARI Analysis Panels

1. **Raw Trajectories** — normalized spatial entropy and max fitness over generations
2. **Rolling Window** — smoothed entropy reduction vs. smoothed best fitness (the two signals whose correlation forms ρ in the ARI formula)
3. **Delta Scatter** — per-generation entropy change vs. fitness change, classified into four quadrants: Construction (H↓ F↑), Degradation (H↑ F↑), Waste (H↓ F↓), Destruction (H↑ F↓)
4. **Rolling Scatter** — rolling signals plotted against each other, colored by generation
5. **Rolling Delta Correlation** — local ARI over a sliding window, showing phases of constructive vs. destructive agency
6. **Concordance** — bar chart of generation event classification counts

### Console Output

Each generation prints fitness statistics, entropy, and drop telemetry:
```
Gen   42 | best=1234.56 mean=89.12 | entropy=0.891234
  [drops] DR_in_paths=150 attempted=23 succeeded=18 items_picked=312
```

The drop telemetry tracks how many DR genes appear in paths, how many drop attempts occur (requires a non-empty sack), and how many succeed (requires the target position to be unoccupied).

---

## MEGA as Baseline

MEGA is not an LLM. It is an evolutionary algorithm that operates over a vocabulary of seven action tokens (R, L, U, D, F, B, DR) and evolves variable-length action sequences through mutation alone (crossover is disabled in the benchmark configuration). Its distinguishing feature is *mutable encoding*: it can compress frequently-used subsequences into metagenes — nested, reusable action patterns — through evolutionary pressure, without any explicit prompting or language capability.

MEGA's stochastic nature means performance varies across runs. In the results reported in the writeup, five runs produced ARI values ranging from 0.000459 to 0.028662 and best fitness from 3,441 to 53,694. This variance is informative: it demonstrates that the benchmark captures real behavioral differences rather than artifacts, and that even the evolutionary baseline can collapse to the step-reward plateau when it fails to discover the deeper mechanics.

---

## Relationship to the Kaggle Benchmark

The Kaggle Benchmark task uses the same environment (`TSS_benchmark.py`) but wraps it in the `kaggle-benchmarks` SDK to evaluate LLMs. Over 70 generations, each model receives fitness scores from the previous generation and must output 100 action paths as JSON arrays. A sliding window of 5 generations provides the only history — forcing models to encode learned knowledge into evolving hypotheses.

This repo provides the environment source and the MEGA baseline. The Kaggle task handles LLM prompting, response parsing, and scoring.

---

## Design Notes: What the Prompt Does Not Tell the Model

The benchmark prompt deliberately withholds information about the X (drop) action. Models are told only that X is "context-dependent — its effects on the environment may not be immediate or obvious." They are not told:

- That X does not count as a step and consumes no step-reward budget.
- That X's immediate reward (0.001) is negligible — and that the real immediate effect of a drop is *negative*: the agent loses the dropped item's value and reduces the exponential multiplier (1.75^n), costing far more than the 0.001 gained.
- That X's real effects are structural — reshaping the item distribution for subsequent paths — rather than immediate.
- That successful drops require an unoccupied target cell.

This ambiguity is load-bearing for the benchmark, not an oversight. Disclosing X's mechanics would collapse three distinct cognitive demands into a single instruction-following task:

1. **Discovery** — inferring that X is a meaningful action class rather than filler, by attending to patterns in the noisy fitness signal. This is hard precisely because X's immediate effect on fitness is negative — the agent must look past the current-path cost to detect the delayed structural benefit.
2. **Cost-benefit reasoning under uncertainty** — recognizing that each drop is an immediate sacrifice (lost item value and multiplier) for a speculative future gain that depends on subsequent paths being adapted to the reshaped space.
3. **Planning under delayed feedback** — recognizing that X's payoff is realized through subsequent traversals of reshaped space, not through the drop itself, and selecting drops that reshape the environment in a direction the agent can exploit.

If models were told "X is free, use it strategically to restructure the environment," they would no longer need to discover X's action class, form a correct causal model of its costs and benefits, or reason about delayed structural payoffs. The benchmark would reduce to an execution task — how well does the model follow spatial-packing instructions — rather than a test of whether the model can construct and maintain its own model of an unfamiliar action's role in its environment.

The discriminative power of ARI depends on this. In the results reported in the writeup, Claude Opus 4.6 drops aggressively (419 drops across 56 generations, highest displacement of any LLM) but with weak directional alignment (ρ = 0.18). GPT-5.4-mini drops selectively (81 drops across 25 generations) with strong alignment (ρ = 0.53). DeepSeek and Gemma show negative ARI, indicating destructive restructuring. These are qualitatively different failure modes of the same underlying task — and they only emerge as distinct signatures because each model has to construct its own causal theory of X from behavior. If the prompt told them what X does, the distinction would wash out.

One consequence worth acknowledging: a model that is cautious by disposition may underuse X entirely, registering near-zero ARI not from incoherent restructuring but from insufficient exploration. This is not a confounder — underexploration of a novel action class is itself a legitimate executive function failure (cognitive flexibility deficit) — but it means near-zero ARI should be interpreted as "the agent did not exhibit constructive restructuring" rather than "the agent attempted restructuring and failed." The diagnostic panels (delta scatter, concordance chart) distinguish these cases.

---

## Known Limitations

### Missing path-score attribution in LLM feedback

The Kaggle Benchmark task as submitted provides models with per-path fitness scores but **does not include the paths themselves** in the history window. Models see:

```
1: 98.50
2: 342.10
3: 57.80
```

They were intended to see:

```
1: RRUUDFXRRL → 98.50
2: FFLLUUDDXR → 342.10
3: RRRRRRRRUU → 57.80
```

This is a significant limitation. Without seeing their own prior paths alongside the scores those paths produced, the models cannot perform action-level credit assignment. They cannot determine which specific actions, sequences, or drop placements contributed to high or low fitness. The hypothesis-formation loop the prompt requests — "analyze the scores, form hypotheses, test them" — lacks empirical grounding: models can observe that scores changed between generations, but they cannot attribute those changes to anything they actually did.

This undermines the benchmark's path-dependence claim. The environment's core design property is that the order of actions matters — which items are encountered, which are dropped where, which chemistry cascades fire — all depend on the specific path taken. But if the model cannot see its paths, this path-dependent structure is invisible to it. The model receives a distribution of scalar outcomes with no way to trace those outcomes back to the action sequences that produced them.

**What this means for the reported results:** The ARI gradient observed across models (ranging from −0.0009 to +0.0287) is real — the models did produce differentiated restructuring behavior. But the causal mechanism is weaker than intended. Positive ARI in the submitted benchmark likely reflects models whose general strategic reasoning and output variation happened to produce coherent restructuring, not models that learned specific action-outcome mappings from the feedback loop. The executive function demands (planning, inhibition, cognitive flexibility) are still present — the step-reward plateau is still a trap, drops still cost fitness, the environment still changes — but the in-context learning component that was intended to ground those demands in empirical observation of the agent's own behavior is not functioning as designed.

**The fix is one line** in the scored summary construction:

```python
# Current (scores only):
scored_summary = [
    f"{i + 1}: {raw_f:.2f}"
    for i, raw_f in enumerate(raw_fitnesses)
]

# Fixed (paths + scores):
scored_summary = [
    f"{i + 1}: {paths[i]} → {raw_f:.2f}"
    for i, raw_f in enumerate(raw_fitnesses)
]
```

A corrected version with path-score attribution would enable proper credit assignment, grounding the hypothesis-formation loop in observable action-outcome pairs and restoring the path-dependence property that the environment was designed to test.

The environment code, ARI metric, chemistry engine, and MEGA baseline are unaffected by this issue. The limitation is confined to the LLM evaluation prompt in the Kaggle Benchmark task.

---

## Citation

```
Andrews, M. (2026). The Agency Gap: Measuring Constructive Agency Through Environmental
Restructuring. Kaggle Measuring Progress Toward AGI Hackathon.
```

---

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

---

## Author

**Matthew Andrews** — Independent researcher

- GitHub: [ML-flash](https://github.com/ML-flash)
- Email: Matthew.Andrews2024@gmail.com
