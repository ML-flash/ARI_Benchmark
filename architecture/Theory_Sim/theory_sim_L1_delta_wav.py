"""
theory_sim1.py

Layer 1: Capture + Composition Growth + Singularity

Encoding scheme:
0, 1        -> delimiter tokens  [ and ]
2, 3        -> base symbols      0 and 1
4, 5, 6,… -> composition identifiers (assigned sequentially by EA)

Alphabet at generation t = {0,1,2,3} + all composition IDs created so far.
Point mutations and insertions sample uniformly over the full current alphabet.
No fitness function. No selection pressure. Neutral drift only.

Capture:
When a delimiter pair fires capture (probability CAPTURE_PROB, checked when
a delimiter token is visited), the content of the pair is extracted. If:
- content length >= MIN_CAPTURE_LEN (at least 2 tokens), and
- the content tuple is not already in the MG (structural dedup),
then a new composition is created: content is stored in MG, delimiters and
content in the organism are replaced by the new composition identifier.
If content is a duplicate, capture does not fire (no-op for this pair).

MG (Meta-Genome):
A dict mapping composition_id (int) -> tuple of tokens.
Grows monotonically. No decay, no eviction, no recency bias.

EA (Encoding Allocation):
Sequential integer counter starting at 4.
No recycling in Layer 1.

Measurements:

- mg_size(t)          : number of compositions in MG
- base_token_count(t) : total count of tokens with encoding 2 or 3 in population
- comp_token_count(t) : total count of composition identifier tokens in population
- delim_token_count(t): total count of delimiter tokens (0,1) in population
- base_fraction(t)    : base_token_count / (base + comp)  [ignoring delimiters]
- comp_fraction(t)    : comp_token_count / (base + comp)
- singularity         : generation at which comp_fraction first exceeds 0.5

Outputs under ./theory_sim_1/:

- composition_growth.csv
- composition_growth.png   (MG size over time)
- token_fractions.png      (base vs composition token fractions)
- singularity.png          (comp_fraction with singularity marker)
- alphabet_size.png        (total alphabet size = 4 + mg_size)
- contact_dynamics.png     (6-panel contact dynamics with delta overlays)
  """

import os
import csv
import random
import time
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================

POPULATION_SIZE = 500
GENERATIONS     = 600
SEED            = 1

MIN_LEN = 2
MAX_LEN = 100

ENABLE_CROSSOVER = True
NUM_PARENTS      = 200
CROSSOVER_PROB   = 0.50

# Boundary operators
BOUNDARY_INSERT_PROB = 0.0035
BOUNDARY_REMOVE_PROB = 0.0010

# Capture probability (checked per delimiter token visit, same scan as mutation)
CAPTURE_PROB = 0.10

# Minimum content length for a valid capture
MIN_CAPTURE_LEN = 2

# Basic mutation rates
MUTATION_PROB           = 0.05   # depth-0 tokens
DELIMITED_MUTATION_PROB = 0.02   # interior tokens

PROGRESS_EVERY = 20

OUTPUT_FOLDER = "theory_sim_L1_delta_wav"
SOUND_FOLDER  = "theory_sim_L1_delta_wav/sound_data"

# Generation at which to start the contact dynamics plot and audio export.
# Generations 0-9 are the structural bootstrapping phase — the transition
# from flat sequences to delimiter-structured organisms. This is a distinct
# event from the evolutionary dynamics and is excluded from the dynamics view.
DYNAMICS_START_GEN = 10

# Encoding constants
ENC_OPEN  = 0   # [
ENC_CLOSE = 1   # ]
ENC_BASE0 = 2   # base symbol 0
ENC_BASE1 = 3   # base symbol 1
FIRST_COMP_ID = 4

# =========================
# META-GENOME + EA
# =========================

class MetaGenome:
    def __init__(self):
        self.store   = {}
        self.reverse = {}
        self.next_id = FIRST_COMP_ID

    def try_capture(self, content_tuple):
        if len(content_tuple) < MIN_CAPTURE_LEN:
            return None, False
        if content_tuple in self.reverse:
            return None, False
        cid = self.next_id
        self.next_id += 1
        self.store[cid] = content_tuple
        self.reverse[content_tuple] = cid
        return cid, True

    def size(self):
        return len(self.store)

    def all_ids(self):
        return list(self.store.keys())

# =========================
# ALPHABET
# =========================

def full_alphabet(mg):
    return [ENC_BASE0, ENC_BASE1] + mg.all_ids()

def random_non_delimiter(rng, mg):
    alphabet = full_alphabet(mg)
    return rng.choice(alphabet)

# =========================
# INITIALIZATION
# =========================

def init_population(rng):
    pop = []
    for _ in range(POPULATION_SIZE):
        L = rng.randint(MIN_LEN, MAX_LEN)
        pop.append([rng.choice([ENC_BASE0, ENC_BASE1]) for _ in range(L)])
    return pop

# =========================
# STRUCTURE HELPERS
# =========================

def is_delimiter(tok):
    return tok == ENC_OPEN or tok == ENC_CLOSE

def is_base(tok):
    return tok == ENC_BASE0 or tok == ENC_BASE1

def is_comp(tok):
    return tok >= FIRST_COMP_ID

def assert_invariants(org):
    inside = False
    for tok in org:
        if tok == ENC_OPEN:
            if inside:
                raise AssertionError("Nesting detected.")
            inside = True
        elif tok == ENC_CLOSE:
            if not inside:
                raise AssertionError("Unmatched ].")
            inside = False
        else:
            if tok < ENC_BASE0:
                raise AssertionError(f"Unknown token {tok}")
    if inside:
        raise AssertionError("Unmatched [ at end.")

def rescan_inside(org, idx):
    inside = False
    for j in range(min(idx + 1, len(org))):
        if org[j] == ENC_OPEN:
            inside = True
        elif org[j] == ENC_CLOSE:
            inside = False
    return inside

def remove_pair_at(org, idx):
    if org[idx] == ENC_OPEN:
        j = idx + 1
        while j < len(org) and org[j] != ENC_CLOSE:
            j += 1
        if j >= len(org):
            raise AssertionError("Unmatched [ during remove.")
        del org[j]
        del org[idx]
        return max(idx - 1, 0)
    if org[idx] == ENC_CLOSE:
        j = idx - 1
        while j >= 0 and org[j] != ENC_OPEN:
            j -= 1
        if j < 0:
            raise AssertionError("Unmatched ] during remove.")
        del org[idx]
        del org[j]
        return max(j - 1, 0)
    raise AssertionError("remove_pair_at called on non-delimiter.")

# =========================
# CAPTURE
# =========================

def attempt_capture(org, open_idx, mg, rng):
    close_idx = open_idx + 1
    while close_idx < len(org) and org[close_idx] != ENC_CLOSE:
        close_idx += 1
    if close_idx >= len(org):
        return False

    content = tuple(org[open_idx + 1 : close_idx])

    cid, is_new = mg.try_capture(content)
    if not is_new:
        return False

    org[open_idx : close_idx + 1] = [cid]
    return True

# =========================
# SWAP
# =========================

def can_swap(tok_a, tok_b):
    return not (is_delimiter(tok_a) and is_delimiter(tok_b))

def attempt_swap(org, i, rng):
    if len(org) < 2:
        return i, False
    dirs = [1, -1] if rng.random() < 0.5 else [-1, 1]
    for d in dirs:
        j = i + d
        if 0 <= j < len(org) and can_swap(org[i], org[j]):
            org[i], org[j] = org[j], org[i]
            return j, True
    return i, False

# =========================
# MUTATION
# =========================

def mutate_org(org, mg, rng):
    i      = 0
    inside = False

    while i < len(org):
        tok = org[i]

        if is_delimiter(tok):
            if tok == ENC_OPEN:
                inside = True
            else:
                inside = False

            roll = rng.random()

            if roll < BOUNDARY_REMOVE_PROB:
                i = remove_pair_at(org, i)
                inside = rescan_inside(org, i) if org else False
                continue

            if tok == ENC_OPEN and roll < BOUNDARY_REMOVE_PROB + CAPTURE_PROB:
                captured = attempt_capture(org, i, mg, rng)
                if captured:
                    inside = False
                    i += 1
                    continue

            new_i, did_swap = attempt_swap(org, i, rng)
            if did_swap:
                inside = rescan_inside(org, new_i) if org else False
                i = new_i
            i += 1
            continue

        if not inside and rng.random() < BOUNDARY_INSERT_PROB:
            org.insert(i, ENC_OPEN)
            org.insert(i + 2, ENC_CLOSE)
            inside = True
            i += 1
            continue

        rate = DELIMITED_MUTATION_PROB if inside else MUTATION_PROB
        if rng.random() < rate:
            op = rng.choice(["point", "insertion", "deletion", "swap"])

            if op == "point":
                org[i] = random_non_delimiter(rng, mg)
                i += 1
                continue

            if op == "insertion":
                new_tok = random_non_delimiter(rng, mg)
                if rng.random() < 0.5:
                    org.insert(i, new_tok)
                    i += 1
                else:
                    org.insert(i + 1, new_tok)
                    i += 2
                continue

            if op == "deletion":
                if len(org) > MIN_LEN:
                    del org[i]
                    i      = max(i - 1, 0)
                    inside = rescan_inside(org, i)
                    continue

            if op == "swap":
                new_i, did_swap = attempt_swap(org, i, rng)
                if did_swap:
                    inside = rescan_inside(org, new_i)
                    i = new_i
                i += 1
                continue

        i += 1

    assert_invariants(org)

def mutate_population(pop, mg, rng):
    for org in pop:
        mutate_org(org, mg, rng)

# =========================
# STATS
# =========================

def compute_stats(pop, mg):
    delim_count = 0
    base_count  = 0
    comp_count  = 0

    A0 = 0
    A1 = 0
    A2 = 0

    num_pairs  = 0
    seg_sum    = 0
    seg_sq_sum = 0
    seg_max    = 0

    num_gaps  = 0
    gap_sum   = 0
    gap_min   = None
    gaps_eq_0 = 0
    gaps_eq_1 = 0
    gaps_ge_2 = 0
    gaps_le_1 = 0

    for org in pop:
        inside         = False
        seg_len        = 0
        last_block_end = None

        for i, tok in enumerate(org):
            if is_delimiter(tok):
                delim_count += 1
                A2          += 1

                if tok == ENC_OPEN:
                    inside = True
                    if last_block_end is not None:
                        gap = i - (last_block_end + 1)
                        num_gaps += 1
                        gap_sum  += gap
                        if gap_min is None or gap < gap_min:
                            gap_min = gap
                        if gap == 0:
                            gaps_eq_0 += 1
                            gaps_le_1 += 1
                        elif gap == 1:
                            gaps_eq_1 += 1
                            gaps_le_1 += 1
                        else:
                            gaps_ge_2 += 1
                    seg_len = 0

                else:
                    inside     = False
                    num_pairs += 1
                    seg_sum   += seg_len
                    seg_sq_sum += seg_len * seg_len
                    if seg_len > seg_max:
                        seg_max = seg_len
                    last_block_end = i

            else:
                if is_base(tok):
                    base_count += 1
                else:
                    comp_count += 1

                if inside:
                    A1 += 1
                    seg_len += 1
                else:
                    A0 += 1

    non_delim = base_count + comp_count
    base_frac = base_count / non_delim if non_delim > 0 else 0.0
    comp_frac = comp_count / non_delim if non_delim > 0 else 0.0

    mg_size       = mg.size()
    alphabet_size = FIRST_COMP_ID + mg_size

    mean_seg   = (seg_sum    / num_pairs) if num_pairs else 0.0
    seg_var    = ((seg_sq_sum / num_pairs) - mean_seg ** 2) if num_pairs else 0.0
    mean_gap   = (gap_sum    / num_gaps)  if num_gaps  else 0.0
    min_gap    = gap_min if gap_min is not None else 0
    frac_le_1  = (gaps_le_1 / num_gaps)  if num_gaps  else 0.0
    frac_gap_0 = (gaps_eq_0 / num_gaps)  if num_gaps  else 0.0
    frac_gap_1 = (gaps_eq_1 / num_gaps)  if num_gaps  else 0.0
    frac_ge_2  = (gaps_ge_2 / num_gaps)  if num_gaps  else 0.0

    return (
        delim_count, base_count, comp_count, base_frac, comp_frac, mg_size, alphabet_size,
        A0, A1, A2,
        num_pairs, mean_seg, seg_max, seg_var,
        num_gaps, mean_gap, min_gap, frac_le_1,
        frac_gap_0, frac_gap_1, frac_ge_2
    )

# =========================
# CROSSOVER (DEPTH-0 CUTS ONLY)
# =========================

def eligible_cuts_depth0(org):
    cuts   = [0]
    inside = False
    for i, tok in enumerate(org):
        if tok == ENC_OPEN:
            inside = True
        elif tok == ENC_CLOSE:
            inside = False
        if not inside:
            cuts.append(i + 1)
    return cuts

def select_parents(pop, rng):
    return [rng.choice(pop).copy() for _ in range(NUM_PARENTS)]

def make_next_generation(pop, rng):
    if not ENABLE_CROSSOVER:
        return [rng.choice(pop).copy() for _ in range(POPULATION_SIZE)]

    parents  = select_parents(pop, rng)
    next_pop = []

    while len(next_pop) < POPULATION_SIZE:
        a = rng.choice(parents)
        b = rng.choice(parents)

        if rng.random() < CROSSOVER_PROB:
            cuts_a = eligible_cuts_depth0(a)
            cuts_b = eligible_cuts_depth0(b)
            if cuts_a and cuts_b:
                ca     = rng.choice(cuts_a)
                cb     = rng.choice(cuts_b)
                child1 = a[:ca] + b[cb:]
                child2 = b[:cb] + a[ca:]
            else:
                child1 = a.copy()
                child2 = b.copy()
        else:
            child1 = a.copy()
            child2 = b.copy()

        assert_invariants(child1)
        assert_invariants(child2)

        next_pop.append(child1)
        if len(next_pop) < POPULATION_SIZE:
            next_pop.append(child2)

    return next_pop

# =========================
# DELTA HELPER
# =========================

def compute_delta(series):
    """
    Compute per-generation delta (first difference) of a series.
    Returns array of same length; first element is 0.
    """
    arr = np.array(series, dtype=float)
    delta = np.zeros_like(arr)
    delta[1:] = arr[1:] - arr[:-1]
    return delta

def add_delta_overlay(ax, gens, delta, color="black", alpha=0.45, label="Δ"):
    """
    Add delta as a secondary y-axis overlay on an existing axes.
    Returns the secondary axes object.
    """
    ax2 = ax.twinx()
    ax2.plot(gens, delta, color=color, alpha=alpha, linewidth=0.8, linestyle="--", label=label)
    ax2.set_ylabel(label, color=color, fontsize=7)
    ax2.tick_params(axis="y", labelcolor=color, labelsize=6)
    ax2.axhline(0, color=color, linewidth=0.4, alpha=0.3)
    return ax2

# =========================
# MAIN LOOP
# =========================

def run():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(SOUND_FOLDER,  exist_ok=True)

    rng = random.Random(SEED)
    mg  = MetaGenome()
    pop = init_population(rng)

    gens          = []
    delim_counts  = []
    base_counts   = []
    comp_counts   = []
    base_fracs    = []
    comp_fracs    = []
    mg_sizes      = []
    alpha_sizes   = []

    A0_list     = []
    A1_list     = []
    A2_list     = []
    mseg_list   = []
    maxseg_list = []
    segvar_list = []
    mgap_list   = []
    fle1_list   = []
    fgap0_list  = []
    fgap1_list  = []
    fge2_list   = []
    flux_list   = []

    singularity_gen = None
    prev_fgap0      = None

    csv_path = os.path.join(OUTPUT_FOLDER, "composition_growth.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gen", "mg_size", "alphabet_size",
            "delim_count", "base_count", "comp_count",
            "base_frac", "comp_frac",
            "A0", "A1", "A2",
            "num_pairs", "mean_seg", "max_seg", "seg_variance",
            "num_gaps", "mean_gap", "min_gap", "frac_gaps_le1",
            "frac_gap_0", "frac_gap_1", "frac_gap_ge2", "contact_flux"
        ])

        t0 = time.time()

        for gen in range(GENERATIONS + 1):

            (delim_count, base_count, comp_count, base_frac, comp_frac, mg_size, alphabet_size,
             A0, A1, A2,
             num_pairs, mean_seg, seg_max, seg_var,
             num_gaps, mean_gap, min_gap, frac_le_1,
             frac_gap_0, frac_gap_1, frac_ge_2) = compute_stats(pop, mg)

            flux = abs(frac_gap_0 - prev_fgap0) if prev_fgap0 is not None else 0.0
            prev_fgap0 = frac_gap_0

            if singularity_gen is None and comp_frac >= 0.5:
                singularity_gen = gen

            writer.writerow([
                gen, mg_size, alphabet_size,
                delim_count, base_count, comp_count,
                f"{base_frac:.6f}", f"{comp_frac:.6f}",
                A0, A1, A2,
                num_pairs, f"{mean_seg:.4f}", seg_max, f"{seg_var:.4f}",
                num_gaps, f"{mean_gap:.4f}", min_gap, f"{frac_le_1:.4f}",
                f"{frac_gap_0:.4f}", f"{frac_gap_1:.4f}", f"{frac_ge_2:.4f}",
                f"{flux:.4f}"
            ])

            gens.append(gen)
            delim_counts.append(delim_count)
            base_counts.append(base_count)
            comp_counts.append(comp_count)
            base_fracs.append(base_frac)
            comp_fracs.append(comp_frac)
            mg_sizes.append(mg_size)
            alpha_sizes.append(alphabet_size)
            A0_list.append(A0)
            A1_list.append(A1)
            A2_list.append(A2)
            mseg_list.append(mean_seg)
            maxseg_list.append(seg_max)
            segvar_list.append(seg_var)
            mgap_list.append(mean_gap)
            fle1_list.append(frac_le_1)
            fgap0_list.append(frac_gap_0)
            fgap1_list.append(frac_gap_1)
            fge2_list.append(frac_ge_2)
            flux_list.append(flux)

            if gen % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  gen {gen:4d}  mg={mg_size:5d}  "
                      f"base_frac={base_frac:.3f}  comp_frac={comp_frac:.3f}  "
                      f"A0={A0:6d}  A1={A1:6d}  A2={A2:6d}  "
                      f"elapsed={elapsed:.1f}s")

            if gen < GENERATIONS:
                pop = make_next_generation(pop, rng)
                mutate_population(pop, mg, rng)

    if singularity_gen is not None:
        print(f"\n  *** Singularity at generation {singularity_gen} "
              f"(comp_frac crossed 0.5) ***")
    else:
        print(f"\n  Singularity not reached within {GENERATIONS} generations.")

    # ---- Pre-compute deltas for all 6 panel signals ----
    d_fgap0 = compute_delta(fgap0_list)
    d_fgap1 = compute_delta(fgap1_list)
    d_fge2  = compute_delta(fge2_list)
    d_flux  = compute_delta(flux_list)
    d_mgap  = compute_delta(mgap_list)

    # ---- PLOTS ----

    # Boundary geometry
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, A0_list, label="A0 (depth-0 tokens)")
    ax.plot(gens, A1_list, label="A1 (interior tokens)")
    ax.plot(gens, A2_list, label="A2 (boundary tokens)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Token count (population total)")
    ax.set_title("Boundary Geometry Dynamics (Layer 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "boundary_geometry.png"), dpi=150)
    plt.close(fig)

    # Segment lengths
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(gens, mseg_list)
    axes[0].set_title("Mean Segment Length")
    axes[0].set_xlabel("Generation")
    axes[1].plot(gens, maxseg_list)
    axes[1].set_title("Max Segment Length")
    axes[1].set_xlabel("Generation")
    axes[2].plot(gens, segvar_list)
    axes[2].set_title("Segment Length Variance")
    axes[2].set_xlabel("Generation")
    fig.suptitle("Segment Length Dynamics (Layer 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "segment_lengths.png"), dpi=150)
    plt.close(fig)

    # ---- Sound data: export deltas to sound_data/ ----
    # Slice from DYNAMICS_START_GEN — excludes structural bootstrapping phase
    dyn_idx  = next(i for i, g in enumerate(gens) if g >= DYNAMICS_START_GEN)
    dyn_gens = gens[dyn_idx:]

    sound_signals = [
        ("delta_contact_rate",    d_fgap0),
        ("delta_expansion_ready", d_fgap1),
        ("delta_free_space",      d_fge2),
        ("delta_contact_flux",    d_flux),
    ]
    # delta_mean_gap excluded from audio — different physical quantity (spatial extent
    # vs boundary state transition rates). Kept in CSV for reference only.
    mean_gap_path = os.path.join(SOUND_FOLDER, "delta_mean_gap.csv")
    with open(mean_gap_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gen", "delta_mean_gap"])
        for i, g in enumerate(dyn_gens):
            writer.writerow([g, f"{d_mgap[dyn_idx + i]:.6f}"])

    for name, delta in sound_signals:
        path = os.path.join(SOUND_FOLDER, f"{name}.csv")
        dyn_delta = delta[dyn_idx:]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gen", name])
            for i, g in enumerate(dyn_gens):
                writer.writerow([g, f"{dyn_delta[i]:.6f}"])
        print(f"  Sound data: {path}")

    # Combined file for convenience
    combined_path = os.path.join(SOUND_FOLDER, "all_deltas.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gen"] + [name for name, _ in sound_signals])
        for i, g in enumerate(dyn_gens):
            writer.writerow([g] + [f"{delta[dyn_idx + i]:.6f}" for _, delta in sound_signals])
    print(f"  Combined:   {combined_path}")

    # ---- Contact dynamics suite: 2x2 individual panels + wide composite ----
    # Plot and audio both start at DYNAMICS_START_GEN — excludes bootstrapping phase
    delta_panels = [
        (d_fgap0[dyn_idx:], "tab:red",    "Δ Contact Rate (gap=0)"),
        (d_fgap1[dyn_idx:], "tab:orange", "Δ Expansion-Ready Rate (gap=1)"),
        (d_fge2[dyn_idx:],  "tab:green",  "Δ Free Space (gap≥2)"),
        (d_flux[dyn_idx:],  "tab:purple", "Δ Contact Flux"),
    ]

    fig = plt.figure(figsize=(18, 10))
    # 2x2 grid for individual panels (top two rows)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    axes_ind  = []
    for (row, col), (delta, color, title) in zip(positions, delta_panels):
        ax = fig.add_subplot(3, 2, row * 2 + col + 1)
        ax.plot(dyn_gens, delta, color=color, linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Δ per generation")
        axes_ind.append(ax)

    # Wide composite panel spanning full bottom row
    ax_comp = fig.add_subplot(3, 1, 3)
    labels  = ["Δ contact", "Δ exp-ready", "Δ free", "Δ flux"]
    for (delta, color, _), label in zip(delta_panels, labels):
        arr  = np.array(delta)
        peak = np.abs(arr).max()
        norm = arr / peak if peak > 0 else arr
        ax_comp.plot(dyn_gens, norm, color=color, linewidth=0.8, alpha=0.75, label=label)
    ax_comp.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax_comp.set_title("All Δ Signals (per-signal peak)")
    ax_comp.set_xlabel("Generation")
    ax_comp.set_ylabel("Δ / own peak")
    ax_comp.legend(fontsize=8, loc="upper right")

    fig.suptitle("Contact Dynamics — Δ Waveforms (Layer 1)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(SOUND_FOLDER, "contact_dynamics.png"), dpi=150)
    plt.close(fig)

    # Composition library growth
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, mg_sizes, color="tab:purple")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of compositions")
    ax.set_title("Composition Library Growth (Layer 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "composition_growth.png"), dpi=150)
    plt.close(fig)

    # Token type counts
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, base_counts,  label="Base tokens (2,3)",       color="tab:blue")
    ax.plot(gens, comp_counts,  label="Composition tokens (≥4)", color="tab:orange")
    ax.plot(gens, delim_counts, label="Delimiter tokens (0,1)",  color="tab:green", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Token count (population total)")
    ax.set_title("Token Type Distribution (Layer 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "token_counts.png"), dpi=150)
    plt.close(fig)

    # Base vs composition fraction with singularity marker
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, base_fracs, label="Base fraction",        color="tab:blue")
    ax.plot(gens, comp_fracs, label="Composition fraction", color="tab:orange")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 threshold")
    if singularity_gen is not None:
        ax.axvline(singularity_gen, color="red", linestyle=":",
                   linewidth=1.2, label=f"Singularity (gen {singularity_gen})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fraction of non-delimiter tokens")
    ax.set_title("Base vs Composition Token Fractions (Layer 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "token_fractions.png"), dpi=150)
    plt.close(fig)

    # Alphabet size
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, alpha_sizes, color="tab:red")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Alphabet size (4 + compositions)")
    ax.set_title("Total Alphabet Size Over Time (Layer 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "alphabet_size.png"), dpi=150)
    plt.close(fig)

    print(f"\nDone. Outputs in ./{OUTPUT_FOLDER}/")


if __name__ == "__main__":
    run()