"""
theory_sim0.py

Layer 0: Boundary Geometry (Pre-Capture) + Drift + Swap Mobility + Negotiation Metrics

Outputs under ./theory_sim_0/:
- boundary_geometry.csv
- boundary_geometry.png       (A0/A1/A2)
- segment_lengths.png         (mean/max segment length + variance)
- negotiation_gaps.png        (mean gap, frac gaps <=1, frac gaps ==0)
- contact_dynamics.png        (comprehensive negotiation contact suite)

Contact dynamics metrics:
- frac_gap_0    : fraction of boundary pairs at gap=0 (in contact, swap halted)
- frac_gap_1    : fraction at gap=1 (expansion-ready, one G separating boundaries)
- frac_gap_ge2  : fraction at gap>=2 (free, not in negotiation range)
- contact_flux  : abs change in frac_gap_0 per generation (retreat/advance activity)
- seg_variance  : variance in segment length (spread of negotiation outcomes)
"""

import os
import csv
import random
import time
import matplotlib.pyplot as plt


# =========================
# PARAMETERS (EDIT HERE)
# =========================

POPULATION_SIZE = 500
GENERATIONS     = 400
SEED            = 1

MIN_LEN = 2
MAX_LEN = 100

ENABLE_CROSSOVER = True
NUM_PARENTS      = 200
CROSSOVER_PROB   = 0.50

BOUNDARY_INSERT_PROB = 0.0030
BOUNDARY_REMOVE_PROB = 0.0005


MUTATION_PROB           = 0.05   # outside delimiters  (context=0)
DELIMITED_MUTATION_PROB = 0.02   # inside delimiters   (context=1)




PROGRESS_EVERY = 10

OUTPUT_FOLDER       = "theory_sim_0"
OUTPUT_CSV          = "boundary_geometry.csv"
OUTPUT_SURFACE_PLOT = "boundary_geometry.png"
OUTPUT_SEG_PLOT     = "segment_lengths.png"
OUTPUT_GAP_PLOT     = "negotiation_gaps.png"
OUTPUT_CONTACT_PLOT = "contact_dynamics.png"


# =========================
# INITIALIZATION
# =========================

def init_population(rng: random.Random):
    pop = []
    for _ in range(POPULATION_SIZE):
        L = rng.randint(MIN_LEN, MAX_LEN)
        pop.append(["G"] * L)   # initial population: no delimiters (A0 / A8)
    return pop


# =========================
# STRUCTURE HELPERS
# =========================

def assert_invariants_single_pass(org):
    """Validate: no nesting, no unmatched delimiters."""
    inside = False
    for tok in org:
        if tok == "[":
            if inside:
                raise AssertionError("Nesting detected.")
            inside = True
        elif tok == "]":
            if not inside:
                raise AssertionError("Unmatched ']'.")
            inside = False
        else:
            if tok != "G":
                raise AssertionError(f"Unknown token {tok!r}")
    if inside:
        raise AssertionError("Unmatched '['.")


def rescan_inside_state(org, idx):
    """
    Recompute whether position idx is inside a delimiter region.
    Required after any structural change (swap/remove) that may move delimiters.
    """
    inside = False
    for j in range(min(idx + 1, len(org))):
        if org[j] == "[":
            inside = True
        elif org[j] == "]":
            inside = False
    return inside



def remove_pair_at(org, idx):
    """
    Remove the delimiter pair whose opening/closing token is at org[idx].
    Returns a suggested next scan index.
    Paired atomicity: always removes both markers together.
    """
    if org[idx] == "[":
        j = idx + 1
        while j < len(org) and org[j] != "]":
            if org[j] == "[":
                raise AssertionError("Nesting during remove.")
            j += 1
        if j >= len(org):
            raise AssertionError("Unmatched '[' during remove.")
        del org[j]
        del org[idx]
        return max(idx - 1, 0)

    if org[idx] == "]":
        j = idx - 1
        while j >= 0 and org[j] != "[":
            if org[j] == "]":
                raise AssertionError("Nesting during remove.")
            j -= 1
        if j < 0:
            raise AssertionError("Unmatched ']' during remove.")
        del org[idx]
        del org[j]
        return max(j - 1, 0)

    raise AssertionError("remove_pair_at called on non-delimiter.")


# =========================
# SWAP
# =========================

def can_swap(tok_a, tok_b):
    """
    Delimiter<->delimiter swaps are forbidden (A7.3 / MEGA doc).
    Delimiter<->G swaps are permitted -- this is the boundary mobility mechanism.
    """
    return not (tok_a in ("[", "]") and tok_b in ("[", "]"))


def attempt_swap(org, i, rng):
    """
    Try to swap org[i] with a neighbour.
    Randomly picks a direction; falls back to the other if blocked.
    Returns (new_i, swapped: bool).
    """
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

def mutate_org(org, rng: random.Random):
    """
    Single-pass mutation loop with three context branches (PFOL P2):

      context=2  delimiter token   "[" or "]"
        Roll once:
          < BOUNDARY_REMOVE_PROB              -> delete pair (pi6)
          else if can_swap neighbour exists   -> swap  ( (1-pi6)*can_swap )
          else                                -> no-op

      context=0  undelimited G token
        Independent roll BOUNDARY_INSERT_PROB -> insert pair around this G
        Independent roll MUTATION_PROB        -> insertion/deletion/swap

      context=1  delimited G token
        Independent roll DELIMITED_MUTATION_PROB -> insertion/deletion/swap
    """
    i      = 0
    inside = False

    while i < len(org):
        tok = org[i]

        # ---- CONTEXT 2: delimiter token ----
        if tok in ("[", "]"):
            if tok == "[":
                inside = True
            else:
                inside = False

            roll = rng.random()

            if roll < BOUNDARY_REMOVE_PROB:
                i = remove_pair_at(org, i)
                inside = rescan_inside_state(org, i) if org else False
                continue
            else:
                new_i, did_swap = attempt_swap(org, i, rng)
                if did_swap:
                    inside = rescan_inside_state(org, new_i) if org else False
                    i = new_i
                i += 1
                continue

        # ---- CONTEXT 0 / 1: ordinary G token ----
        if tok != "G":
            raise AssertionError(f"Unknown token: {tok!r}")

        # context=0 only: boundary insertion
        if (not inside) and rng.random() < BOUNDARY_INSERT_PROB:
            org.insert(i, "[")
            org.insert(i + 2, "]")
            inside = True
            i += 1
            continue

        # context-specific basic mutation rate
        rate = DELIMITED_MUTATION_PROB if inside else MUTATION_PROB
        if rng.random() < rate:
            op = rng.choice(["insertion", "deletion", "swap"])

            if op == "insertion":
                if rng.random() < 0.5:
                    org.insert(i, "G")
                    i += 1
                else:
                    org.insert(i + 1, "G")
                    i += 2
                continue

            if op == "deletion":
                if len(org) > MIN_LEN:
                    del org[i]
                    i      = max(i - 1, 0)
                    inside = rescan_inside_state(org, i)
                    continue

            if op == "swap":
                new_i, did_swap = attempt_swap(org, i, rng)
                if did_swap:
                    inside = rescan_inside_state(org, new_i)
                    i = new_i
                i += 1
                continue

        i += 1

    assert_invariants_single_pass(org)


def mutate_population(pop, rng):
    for org in pop:
        mutate_org(org, rng)


# =========================
# STATS (ONE PASS)
# =========================

def compute_stats(pop):
    """
    Single-pass per organism.

    Surface areas:   A0, A1, A2
    Segment metrics: num_pairs, mean_seg, seg_max, seg_variance
    Gap distribution (between adjacent boundary pairs):
      num_gaps, mean_gap, min_gap
      frac_gap_0   -- in contact (swap halted; negotiation condition)
      frac_gap_1   -- expansion-ready (one G separating pairs)
      frac_gap_ge2 -- free (not in negotiation range)
      frac_le_1    -- gap_0 + gap_1 combined (legacy metric retained)
    """
    A0 = A1 = A2 = 0

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

        i = 0
        while i < len(org):
            tok = org[i]

            if tok == "[":
                if inside:
                    raise AssertionError("Nesting detected in stats.")
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
                inside  = True
                seg_len = 0
                A2 += 1
                i  += 1
                continue

            if tok == "]":
                if not inside:
                    raise AssertionError("Unmatched ']' in stats.")
                inside     = False
                A2        += 1
                num_pairs += 1
                seg_sum   += seg_len
                seg_sq_sum += seg_len * seg_len
                if seg_len > seg_max:
                    seg_max = seg_len
                last_block_end = i
                i += 1
                continue

            if tok != "G":
                raise AssertionError(f"Unknown token {tok!r} in stats.")

            if inside:
                A1 += 1
                seg_len += 1
            else:
                A0 += 1
            i += 1

        if inside:
            raise AssertionError("Unmatched '[' at end in stats.")

    mean_seg   = (seg_sum / num_pairs) if num_pairs else 0.0
    seg_var    = ((seg_sq_sum / num_pairs) - mean_seg ** 2) if num_pairs else 0.0
    mean_gap   = (gap_sum  / num_gaps) if num_gaps else 0.0
    min_gap    = gap_min if gap_min is not None else 0
    frac_le_1  = (gaps_le_1 / num_gaps) if num_gaps else 0.0
    frac_gap_0 = (gaps_eq_0 / num_gaps) if num_gaps else 0.0
    frac_gap_1 = (gaps_eq_1 / num_gaps) if num_gaps else 0.0
    frac_ge_2  = (gaps_ge_2 / num_gaps) if num_gaps else 0.0

    return (A0, A1, A2,
            num_pairs, mean_seg, seg_max, seg_var,
            num_gaps, mean_gap, min_gap, frac_le_1,
            frac_gap_0, frac_gap_1, frac_ge_2)


# =========================
# CROSSOVER (DEPTH-0 CUTS ONLY)  [PFOL P1]
# =========================

def eligible_cuts_depth0(org):
    cuts   = [0]
    inside = False
    for i, tok in enumerate(org):
        if tok == "[":
            inside = True
        elif tok == "]":
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

        assert_invariants_single_pass(child1)
        assert_invariants_single_pass(child2)

        next_pop.append(child1)
        if len(next_pop) < POPULATION_SIZE:
            next_pop.append(child2)

    return next_pop


# =========================
# MAIN LOOP
# =========================

def run():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    rng = random.Random(SEED)
    pop = init_population(rng)

    gens        = []
    A0_list     = []
    A1_list     = []
    A2_list     = []
    npairs_list = []
    mseg_list   = []
    maxseg_list = []
    segvar_list = []
    ngap_list   = []
    mgap_list   = []
    mingap_list = []
    fle1_list   = []
    fgap0_list  = []
    fgap1_list  = []
    fge2_list   = []
    flux_list   = []

    csv_path = os.path.join(OUTPUT_FOLDER, OUTPUT_CSV)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gen",
            "A0", "A1", "A2",
            "num_pairs", "mean_seg", "max_seg", "seg_variance",
            "num_gaps", "mean_gap", "min_gap",
            "frac_gaps_le1",
            "frac_gap_0", "frac_gap_1", "frac_gap_ge2",
            "contact_flux"
        ])

        t0         = time.time()
        prev_fgap0 = None

        for gen in range(GENERATIONS + 1):

            stats = compute_stats(pop)
            (A0, A1, A2,
             num_pairs, mean_seg, seg_max, seg_var,
             num_gaps, mean_gap, min_gap, frac_le_1,
             frac_gap_0, frac_gap_1, frac_ge_2) = stats

            flux = abs(frac_gap_0 - prev_fgap0) if prev_fgap0 is not None else 0.0
            prev_fgap0 = frac_gap_0

            writer.writerow([
                gen, A0, A1, A2,
                num_pairs, f"{mean_seg:.4f}", seg_max, f"{seg_var:.4f}",
                num_gaps, f"{mean_gap:.4f}", min_gap,
                f"{frac_le_1:.4f}",
                f"{frac_gap_0:.4f}", f"{frac_gap_1:.4f}", f"{frac_ge_2:.4f}",
                f"{flux:.4f}"
            ])

            gens.append(gen)
            A0_list.append(A0);         A1_list.append(A1);      A2_list.append(A2)
            npairs_list.append(num_pairs)
            mseg_list.append(mean_seg); maxseg_list.append(seg_max)
            segvar_list.append(seg_var)
            ngap_list.append(num_gaps); mgap_list.append(mean_gap)
            mingap_list.append(min_gap)
            fle1_list.append(frac_le_1)
            fgap0_list.append(frac_gap_0)
            fgap1_list.append(frac_gap_1)
            fge2_list.append(frac_ge_2)
            flux_list.append(flux)

            if gen % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  gen {gen:4d}  A0={A0:7d}  A1={A1:7d}  A2={A2:7d}  "
                      f"pairs={num_pairs:5d}  gap0={frac_gap_0:.3f}  "
                      f"gap1={frac_gap_1:.3f}  elapsed={elapsed:.1f}s")

            if gen < GENERATIONS:
                pop = make_next_generation(pop, rng)
                mutate_population(pop, rng)

    # ---- PLOTS ----

    # Surface areas
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, A0_list, label="A0 (depth-0 G)")
    ax.plot(gens, A1_list, label="A1 (interior G)")
    ax.plot(gens, A2_list, label="A2 (boundary tokens)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Token count (population total)")
    ax.set_title("Boundary Geometry Dynamics (Layer 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, OUTPUT_SURFACE_PLOT), dpi=150)
    plt.close(fig)

    # Segment lengths + variance
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
    fig.suptitle("Segment Length Dynamics (Layer 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, OUTPUT_SEG_PLOT), dpi=150)
    plt.close(fig)

    # Legacy negotiation gaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(gens, mgap_list)
    axes[0].set_title("Mean Gap Length")
    axes[0].set_xlabel("Generation")
    axes[1].plot(gens, fle1_list)
    axes[1].set_title("Fraction Gaps <= 1")
    axes[1].set_xlabel("Generation")
    axes[2].plot(gens, fgap0_list)
    axes[2].set_title("Fraction Gaps = 0")
    axes[2].set_xlabel("Generation")
    fig.suptitle("Boundary Negotiation Metrics (Layer 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, OUTPUT_GAP_PLOT), dpi=150)
    plt.close(fig)

    # Contact dynamics suite
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Gap state distribution
    axes[0, 0].plot(gens, fgap0_list, label="gap=0 (contact)")
    axes[0, 0].plot(gens, fgap1_list, label="gap=1 (expansion-ready)")
    axes[0, 0].plot(gens, fge2_list,  label="gap>=2 (free)")
    axes[0, 0].set_title("Gap State Distribution")
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Fraction of gaps")
    axes[0, 0].legend(fontsize=8)

    # Contact rate
    axes[0, 1].plot(gens, fgap0_list, color="tab:red")
    axes[0, 1].set_title("Contact Rate (gap=0)")
    axes[0, 1].set_xlabel("Generation")
    axes[0, 1].set_ylabel("Fraction of boundary pairs in contact")

    # Expansion-ready rate
    axes[0, 2].plot(gens, fgap1_list, color="tab:orange")
    axes[0, 2].set_title("Expansion-Ready Rate (gap=1)")
    axes[0, 2].set_xlabel("Generation")
    axes[0, 2].set_ylabel("Fraction of boundary pairs at gap=1")

    # Contact flux
    axes[1, 0].plot(gens, flux_list, color="tab:purple", alpha=0.7)
    axes[1, 0].set_title("Contact Flux (|delta gap=0| per generation)")
    axes[1, 0].set_xlabel("Generation")
    axes[1, 0].set_ylabel("Abs change in contact fraction")

    # Mean gap vs contact rate overlay
    ax2 = axes[1, 1].twinx()
    axes[1, 1].plot(gens, mgap_list, color="tab:blue", label="mean gap")
    ax2.plot(gens, fgap0_list, color="tab:red", alpha=0.6, label="contact rate")
    axes[1, 1].set_title("Mean Gap vs Contact Rate")
    axes[1, 1].set_xlabel("Generation")
    axes[1, 1].set_ylabel("Mean gap length", color="tab:blue")
    ax2.set_ylabel("Contact fraction", color="tab:red")
    axes[1, 1].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # Segment variance
    axes[1, 2].plot(gens, segvar_list, color="tab:green")
    axes[1, 2].set_title("Segment Length Variance")
    axes[1, 2].set_xlabel("Generation")
    axes[1, 2].set_ylabel("Variance")

    fig.suptitle("Contact Dynamics Suite (Layer 0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, OUTPUT_CONTACT_PLOT), dpi=150)
    plt.close(fig)

    print(f"\nDone. Outputs in ./{OUTPUT_FOLDER}/")


if __name__ == "__main__":
    run()