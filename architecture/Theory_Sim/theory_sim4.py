"""
theory_sim4.py

Layer 4: Meta-gene Creation Order (MCO) — Recency-Biased Selection

Layer 0, 1, 2, 3 (carried forward):
  Boundary geometry dynamics, MG, EA, DAG instrumentation,
  COMP_CHOICE_PROB, and OPEN_PROB all preserved exactly as implemented
  in Layer 3.

Layer 4 addition:
  MCO_DECAY (π₁₃) controls recency-biased composition selection during
  mutation. The MCO is an ordered stack: new captures append to the top.
  When a mutation event selects a composition token, the probability of
  selecting meta-gene j (0-indexed from oldest) is:

      p(mⱼ) ∝ π₁₃^(|M| - j - 1)

  Values of MCO_DECAY near 1.0 recover near-uniform selection (Layer 3
  behavior). Values near 0.0 concentrate selection on the most recently
  captured compositions, driving deep vertical hierarchy at the expense
  of breadth.

  Predicted dynamics:
    - Cleaner, more deterministic approach to singularity vs Layer 3's
      noisy pressure dynamics.
    - Faster DAG depth growth as new compositions build on recent captures.
    - Accelerated orphan accumulation as older compositions fall outside
      the effective selection window permanently.
    - The singularity returns with force: MCO doesn't oppose collapse,
      it organizes it.

  New Telemetry:
    - MCO Selection Entropy: Shannon entropy of the decay distribution.
    - MCO Effective Window: Compositions holding ≥90% selection mass.
    - MCO Dominant Rank: Position of dominant composition in MCO stack.
    - MCO Top-5 Population Fraction: Population share held by the 5
      most recent compositions.

Outputs under ./theory_sim_4_ccp{}_op{}_mco{}/:
  (All Layer 3 plots)
  + layer4_mco_selection_dynamics.png
  + layer4_mco_vs_singularity.png
"""

import os
import csv
import random
import math
import time
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================

POPULATION_SIZE = 500
GENERATIONS     = 8000
SEED            = 1

MIN_LEN = 2
MAX_LEN = 100

ENABLE_CROSSOVER = True
NUM_PARENTS      = 200
CROSSOVER_PROB   = 0.50

# Boundary operators
BOUNDARY_INSERT_PROB = 0.0035
BOUNDARY_REMOVE_PROB = 0.0020

# Capture probability
CAPTURE_PROB = 0.09
MIN_CAPTURE_LEN = 2

# Basic mutation rates
MUTATION_PROB           = 0.10
DELIMITED_MUTATION_PROB = 0.05

# Layer 3 Parameter: Outward Force
OPEN_PROB = 0.000

PROGRESS_EVERY = 20
SINGULARITY_THRESHOLD = 0.95

# Layer 2 Parameter
COMP_CHOICE_PROB = 0.47

# Layer 4 Parameter: MCO Decay (π₁₃)
MCO_DECAY = 0.50

# Encoding constants
ENC_OPEN      = 0
ENC_CLOSE     = 1
ENC_BASE0     = 2
ENC_BASE1     = 3
FIRST_COMP_ID = 4

OUTPUT_FOLDER = f"theory_sim_4_ccp{COMP_CHOICE_PROB:.2f}_op{OPEN_PROB:.2f}_mco{MCO_DECAY:.2f}"


# =========================
# META-GENOME + EA + MCO
# =========================

class MetaGenome:
    def __init__(self):
        self.store   = {}
        self.reverse = {}
        self.next_id = FIRST_COMP_ID
        self.deps    = {}
        self.rdeps   = {}
        # Layer 4: MCO stack — ordered list, oldest first, newest at end
        self.mco     = []

    def try_capture(self, content_tuple):
        if len(content_tuple) < MIN_CAPTURE_LEN:
            return None, False
        if content_tuple in self.reverse:
            return None, False
        cid = self.next_id
        self.next_id += 1
        self.store[cid]             = content_tuple
        self.reverse[content_tuple] = cid

        referenced = set(tok for tok in content_tuple if tok >= FIRST_COMP_ID)
        self.deps[cid] = referenced
        self.rdeps.setdefault(cid, set())
        for ref in referenced:
            self.rdeps.setdefault(ref, set()).add(cid)

        # Layer 4: Append to top of MCO stack
        self.mco.append(cid)

        return cid, True

    def size(self):
        return len(self.store)

    def all_ids(self):
        return list(self.store.keys())

    def sample_composition(self, rng):
        """Layer 4: Sample a composition ID from the MCO stack using
        exponential decay. p(mⱼ) ∝ π₁₃^(|M| - j - 1) where j is
        0-indexed from oldest."""
        n = len(self.mco)
        if n == 0:
            return None

        # Build unnormalized weights: oldest (j=0) gets decay^(n-1),
        # newest (j=n-1) gets decay^0 = 1.0
        weights = []
        for j in range(n):
            weights.append(MCO_DECAY ** (n - j - 1))

        # Weighted selection
        total = sum(weights)
        roll = rng.random() * total
        cumulative = 0.0
        for j in range(n):
            cumulative += weights[j]
            if roll <= cumulative:
                return self.mco[j]
        # Fallback (float precision)
        return self.mco[-1]

    def mco_entropy(self):
        """Shannon entropy of the MCO selection distribution."""
        n = len(self.mco)
        if n <= 1:
            return 0.0
        weights = [MCO_DECAY ** (n - j - 1) for j in range(n)]
        total = sum(weights)
        entropy = 0.0
        for w in weights:
            p = w / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def mco_effective_window(self):
        """Number of compositions (from top of stack) holding ≥90% of
        cumulative selection probability."""
        n = len(self.mco)
        if n == 0:
            return 0
        # Weights from newest to oldest for cumulative sum from top
        weights_newest_first = [MCO_DECAY ** k for k in range(n)]
        total = sum(weights_newest_first)
        cumulative = 0.0
        for count, w in enumerate(weights_newest_first, 1):
            cumulative += w / total
            if cumulative >= 0.90:
                return count
        return n

    def mco_rank_of(self, cid):
        """Rank of composition in MCO stack, 0 = newest (top).
        Returns -1 if not found."""
        n = len(self.mco)
        for j in range(n - 1, -1, -1):
            if self.mco[j] == cid:
                return n - 1 - j
        return -1

    def dag_metrics(self):
        ids = list(self.store.keys())
        n   = len(ids)
        if n == 0:
            return 0, 0, 0, 0.0, 0, 0

        edges = sum(len(self.deps[i]) for i in ids)
        roots = sum(1 for i in ids if not self.rdeps.get(i))

        depth_cache = {}

        def depth(nid, visiting=None):
            if nid in depth_cache:
                return depth_cache[nid]
            if visiting is None:
                visiting = set()
            if nid in visiting:
                return 0
            visiting.add(nid)
            d = self.deps.get(nid, set())
            result = 0 if not d else 1 + max(depth(dep, visiting) for dep in d)
            depth_cache[nid] = result
            return result

        depths     = [depth(i) for i in ids]
        max_depth  = max(depths) if depths else 0
        mean_depth = sum(depths) / len(depths) if depths else 0.0

        adj = {i: set() for i in ids}
        for i in ids:
            for j in self.deps.get(i, set()):
                if j in adj:
                    adj[i].add(j)
                    adj[j].add(i)

        visited = set()
        lcc     = 0

        def bfs(start):
            queue = [start]
            visited.add(start)
            count = 0
            while queue:
                node = queue.pop()
                count += 1
                for nb in adj.get(node, set()):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            return count

        for i in ids:
            if i not in visited:
                comp_size = bfs(i)
                if comp_size > lcc:
                    lcc = comp_size

        return n, edges, max_depth, mean_depth, roots, lcc


# =========================
# ALPHABET / TOKEN SAMPLING
# =========================

def sample_token(rng, mg):
    if mg.size() > 0 and rng.random() < COMP_CHOICE_PROB:
        # Layer 4: MCO-weighted selection replaces uniform
        return mg.sample_composition(rng)
    return rng.choice([ENC_BASE0, ENC_BASE1])


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

def attempt_capture(org, open_idx, mg, rng, events):
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
    events["captures"] += 1
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

def mutate_org(org, mg, rng, events):
    i      = 0
    inside = False

    while i < len(org):
        tok = org[i]

        if is_delimiter(tok):
            inside = (tok == ENC_OPEN)
            roll = rng.random()

            if roll < BOUNDARY_REMOVE_PROB:
                i = remove_pair_at(org, i)
                inside = rescan_inside(org, i) if org else False
                continue

            if tok == ENC_OPEN and roll < BOUNDARY_REMOVE_PROB + CAPTURE_PROB:
                captured = attempt_capture(org, i, mg, rng, events)
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
            events["baseline_bounds"] += 1
            continue

        rate = DELIMITED_MUTATION_PROB if inside else MUTATION_PROB

        # --- LAYER 3: OPEN MUTATION ---
        if is_comp(tok) and rng.random() < OPEN_PROB:
            content = list(mg.store[tok])
            if not inside:
                expanded = [ENC_OPEN] + content + [ENC_CLOSE]
                events["open_bounds"] += 1
            else:
                expanded = content

            org[i : i+1] = expanded
            i += len(expanded)
            events["opens"] += 1
            continue

        if rng.random() < rate:
            op = rng.choice(["point", "insertion", "deletion", "swap"])

            if op == "point":
                org[i] = sample_token(rng, mg)
                i += 1
                continue

            if op == "insertion":
                new_tok = sample_token(rng, mg)
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


def mutate_population(pop, mg, rng, events):
    for org in pop:
        mutate_org(org, mg, rng, events)


# =========================
# STATS
# =========================

def compute_stats(pop, mg):
    delim_count = 0
    base_count  = 0
    comp_count  = 0
    A0 = A1 = A2 = 0

    num_pairs  = 0
    seg_sum    = seg_sq_sum = seg_max = 0
    num_gaps   = gap_sum = 0
    gap_min    = None
    gaps_eq_0 = gaps_eq_1 = gaps_ge_2 = gaps_le_1 = 0

    comp_freq = {}

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
                            gaps_eq_0 += 1; gaps_le_1 += 1
                        elif gap == 1:
                            gaps_eq_1 += 1; gaps_le_1 += 1
                        else:
                            gaps_ge_2 += 1
                    seg_len = 0
                else:
                    inside = False
                    num_pairs  += 1
                    seg_sum    += seg_len
                    seg_sq_sum += seg_len * seg_len
                    if seg_len > seg_max:
                        seg_max = seg_len
                    last_block_end = i
            else:
                if is_base(tok):
                    base_count += 1
                else:
                    comp_count += 1
                    comp_freq[tok] = comp_freq.get(tok, 0) + 1
                if inside:
                    A1 += 1; seg_len += 1
                else:
                    A0 += 1

    non_delim  = base_count + comp_count
    base_frac  = base_count / non_delim if non_delim else 0.0
    comp_frac  = comp_count / non_delim if non_delim else 0.0
    mg_size    = mg.size()
    alpha_size = FIRST_COMP_ID + mg_size

    mean_seg   = seg_sum    / num_pairs if num_pairs else 0.0
    seg_var    = (seg_sq_sum / num_pairs - mean_seg ** 2) if num_pairs else 0.0
    mean_gap   = gap_sum    / num_gaps  if num_gaps  else 0.0
    min_gap    = gap_min if gap_min is not None else 0
    frac_le_1  = gaps_le_1 / num_gaps  if num_gaps  else 0.0
    frac_gap_0 = gaps_eq_0 / num_gaps  if num_gaps  else 0.0
    frac_gap_1 = gaps_eq_1 / num_gaps  if num_gaps  else 0.0
    frac_ge_2  = gaps_ge_2 / num_gaps  if num_gaps  else 0.0

    if comp_freq:
        dom_id   = max(comp_freq, key=comp_freq.get)
        dom_frac = comp_freq[dom_id] / non_delim if non_delim else 0.0
    else:
        dom_id   = -1
        dom_frac = 0.0

    # Layer 3: Orphan Tracking
    orphans = sum(1 for cid in mg.all_ids() if cid not in comp_freq)
    orphan_frac = orphans / mg_size if mg_size > 0 else 0.0

    # Layer 4: MCO metrics
    mco_ent = mg.mco_entropy()
    mco_win = mg.mco_effective_window()
    mco_dom_rank = mg.mco_rank_of(dom_id) if dom_id >= FIRST_COMP_ID else -1

    # Layer 4: Top-5 MCO population fraction
    top5_ids = set(mg.mco[-5:]) if len(mg.mco) >= 5 else set(mg.mco)
    top5_count = sum(comp_freq.get(cid, 0) for cid in top5_ids)
    mco_top5_frac = top5_count / non_delim if non_delim else 0.0

    return (
        delim_count, base_count, comp_count, base_frac, comp_frac, mg_size, alpha_size,
        A0, A1, A2,
        num_pairs, mean_seg, seg_max, seg_var,
        num_gaps, mean_gap, min_gap, frac_le_1,
        frac_gap_0, frac_gap_1, frac_ge_2,
        dom_id, dom_frac,
        orphans, orphan_frac,
        mco_ent, mco_win, mco_dom_rank, mco_top5_frac
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
# MAIN LOOP
# =========================

def run():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    rng = random.Random(SEED)
    mg  = MetaGenome()
    pop = init_population(rng)

    param_label = f"ccp={COMP_CHOICE_PROB:.2f} op={OPEN_PROB:.2f} mco={MCO_DECAY:.2f}"

    gens          = []
    delim_counts  = []
    base_counts   = []
    comp_counts   = []
    base_fracs    = []
    comp_fracs    = []
    mg_sizes      = []
    alpha_sizes   = []
    A0_list       = []
    A1_list       = []
    A2_list       = []
    mseg_list     = []
    maxseg_list   = []
    segvar_list   = []
    mgap_list     = []
    fle1_list     = []
    fgap0_list    = []
    fgap1_list    = []
    fge2_list     = []
    flux_list     = []
    dag_nodes_list     = []
    dag_edges_list     = []
    dag_maxdepth_list  = []
    dag_meandepth_list = []
    dag_roots_list     = []
    dag_lcc_list       = []
    dom_frac_list      = []

    # Layer 3 Lists
    captures_list     = []
    opens_list        = []
    base_bounds_list  = []
    open_bounds_list  = []
    orphans_list      = []
    orphan_frac_list  = []

    # Layer 4 Lists
    mco_entropy_list     = []
    mco_window_list      = []
    mco_dom_rank_list    = []
    mco_top5_frac_list   = []

    singularity_gen       = None
    comp_exceeds_base_gen = None
    prev_fgap0            = None

    events = {"captures": 0, "opens": 0, "baseline_bounds": 0, "open_bounds": 0}

    csv_path = os.path.join(OUTPUT_FOLDER, "composition_growth_layer4.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gen", "mg_size", "alphabet_size",
            "delim_count", "base_count", "comp_count",
            "base_frac", "comp_frac",
            "A0", "A1", "A2",
            "num_pairs", "mean_seg", "max_seg", "seg_variance",
            "num_gaps", "mean_gap", "min_gap", "frac_gaps_le1",
            "frac_gap_0", "frac_gap_1", "frac_gap_ge2", "contact_flux",
            "dag_nodes", "dag_edges", "dag_max_depth", "dag_mean_depth",
            "dag_roots", "dag_lcc",
            "dominant_comp_id", "dominant_comp_frac",
            "captures", "opens", "baseline_bounds", "open_bounds",
            "orphans", "orphan_frac",
            "mco_entropy", "mco_effective_window",
            "mco_dominant_rank", "mco_top5_pop_frac"
        ])

        t0 = time.time()

        for gen in range(GENERATIONS + 1):

            (delim_count, base_count, comp_count, base_frac, comp_frac, mg_size, alpha_size,
             A0, A1, A2,
             num_pairs, mean_seg, seg_max, seg_var,
             num_gaps, mean_gap, min_gap, frac_le_1,
             frac_gap_0, frac_gap_1, frac_ge_2,
             dom_id, dom_frac,
             orphans, orphan_frac,
             mco_ent, mco_win, mco_dom_rank, mco_top5_frac) = compute_stats(pop, mg)

            flux = abs(frac_gap_0 - prev_fgap0) if prev_fgap0 is not None else 0.0
            prev_fgap0 = frac_gap_0

            if comp_exceeds_base_gen is None and comp_frac >= 0.5:
                comp_exceeds_base_gen = gen

            if singularity_gen is None and dom_frac >= SINGULARITY_THRESHOLD:
                singularity_gen = gen

            dag_n, dag_e, dag_md, dag_mnd, dag_r, dag_lcc = mg.dag_metrics()

            writer.writerow([
                gen, mg_size, alpha_size,
                delim_count, base_count, comp_count,
                f"{base_frac:.6f}", f"{comp_frac:.6f}",
                A0, A1, A2,
                num_pairs, f"{mean_seg:.4f}", seg_max, f"{seg_var:.4f}",
                num_gaps, f"{mean_gap:.4f}", min_gap, f"{frac_le_1:.4f}",
                f"{frac_gap_0:.4f}", f"{frac_gap_1:.4f}", f"{frac_ge_2:.4f}",
                f"{flux:.4f}",
                dag_n, dag_e, dag_md, f"{dag_mnd:.4f}", dag_r, dag_lcc,
                dom_id, f"{dom_frac:.6f}",
                events["captures"], events["opens"],
                events["baseline_bounds"], events["open_bounds"],
                orphans, f"{orphan_frac:.6f}",
                f"{mco_ent:.6f}", mco_win,
                mco_dom_rank, f"{mco_top5_frac:.6f}"
            ])

            gens.append(gen)
            delim_counts.append(delim_count); base_counts.append(base_count); comp_counts.append(comp_count)
            base_fracs.append(base_frac); comp_fracs.append(comp_frac)
            mg_sizes.append(mg_size); alpha_sizes.append(alpha_size)
            A0_list.append(A0); A1_list.append(A1); A2_list.append(A2)
            mseg_list.append(mean_seg); maxseg_list.append(seg_max); segvar_list.append(seg_var)
            mgap_list.append(mean_gap); fle1_list.append(frac_le_1)
            fgap0_list.append(frac_gap_0); fgap1_list.append(frac_gap_1); fge2_list.append(frac_ge_2)
            flux_list.append(flux)
            dag_nodes_list.append(dag_n); dag_edges_list.append(dag_e)
            dag_maxdepth_list.append(dag_md); dag_meandepth_list.append(dag_mnd)
            dag_roots_list.append(dag_r); dag_lcc_list.append(dag_lcc)
            dom_frac_list.append(dom_frac)

            # Layer 3 telemetry
            captures_list.append(events["captures"])
            opens_list.append(events["opens"])
            base_bounds_list.append(events["baseline_bounds"])
            open_bounds_list.append(events["open_bounds"])
            orphans_list.append(orphans)
            orphan_frac_list.append(orphan_frac)

            # Layer 4 telemetry
            mco_entropy_list.append(mco_ent)
            mco_window_list.append(mco_win)
            mco_dom_rank_list.append(mco_dom_rank)
            mco_top5_frac_list.append(mco_top5_frac)

            if gen % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  gen {gen:4d}  mg={mg_size:5d}  "
                      f"base={base_frac:.3f}  comp={comp_frac:.3f}  "
                      f"orphans={orphan_frac:.2%}  dag_depth={dag_md:3d}  "
                      f"mco_win={mco_win:4d}  top5={mco_top5_frac:.3f}  "
                      f"elapsed={elapsed:.1f}s")

            # Reset events for the upcoming generation's mutation phase
            events = {"captures": 0, "opens": 0, "baseline_bounds": 0, "open_bounds": 0}

            if gen < GENERATIONS:
                pop = make_next_generation(pop, rng)
                mutate_population(pop, mg, rng, events)

    if singularity_gen is not None:
        print(f"\n  *** Singularity at generation {singularity_gen} ***")
    else:
        print(f"\n  Singularity not reached within {GENERATIONS} generations.")

    # ================================================================
    # PLOTS
    # ================================================================

    sg  = singularity_gen
    sg2 = comp_exceeds_base_gen

    def mark_singularity(ax):
        if sg2 is not None:
            ax.axvline(sg2, color="orange", linestyle="--", linewidth=0.9, label=f"Comps > base (gen {sg2})")
        if sg is not None:
            ax.axvline(sg, color="red", linestyle=":", linewidth=1.2, label=f"Singularity (gen {sg})")

    def title(base):
        return f"{base}  [{param_label}]"

    # -------------------------------------------------------------
    # NEW LAYER 4 PLOTS
    # -------------------------------------------------------------

    # 1. MCO Selection Dynamics (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Selection Entropy
    axes[0].plot(gens, mco_entropy_list, color="tab:blue")
    mark_singularity(axes[0])
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Shannon Entropy (bits)")
    axes[0].set_title(title("MCO Selection Entropy"))

    # Panel 2: Effective Window vs MG Size
    axes[1].plot(gens, mco_window_list, color="tab:green", label="Effective Window (90%)")
    axes[1].plot(gens, mg_sizes, color="tab:purple", linestyle="--", alpha=0.6, label="MG Size")
    mark_singularity(axes[1])
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Count")
    axes[1].set_title(title("Effective Selection Window vs MG Size"))
    axes[1].legend(fontsize=8)

    # Panel 3: Top-5 MCO Population Fraction
    axes[2].plot(gens, mco_top5_frac_list, color="tab:orange")
    mark_singularity(axes[2])
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Fraction of non-delimiter tokens")
    axes[2].set_title(title("Top-5 MCO Population Fraction"))

    fig.suptitle(title("MCO Selection Dynamics"), fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer4_mco_selection_dynamics.png"), dpi=150)
    plt.close(fig)

    # 2. MCO vs Singularity
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(gens, dom_frac_list, color="tab:red", linewidth=1.5, label="Dominant Comp Fraction")
    ax1.axhline(SINGULARITY_THRESHOLD, color="darkred", linestyle="--", linewidth=0.8,
                label=f"Singularity ({SINGULARITY_THRESHOLD:.0%})")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Dominant Comp Fraction", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2.plot(gens, mco_dom_rank_list, color="tab:blue", alpha=0.7, label="Dominant Comp MCO Rank")
    ax2.set_ylabel("MCO Rank (0 = newest)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    mark_singularity(ax1)
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle(title("MCO Rank vs Singularity Formation"), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer4_mco_vs_singularity.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------
    # LAYER 3 PRESERVED PLOTS
    # -------------------------------------------------------------

    # Capture vs Open Flux
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, captures_list, color="tab:red", alpha=0.8, label="Successful Captures (Inward Force)")
    ax.plot(gens, opens_list, color="tab:blue", alpha=0.8, label="Successful Opens (Outward Force)")
    mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("Events per Generation")
    ax.set_title(title("Thermodynamic Pressure (Capture vs. Open Flux)"))
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer3_pressure_flux.png"), dpi=150)
    plt.close(fig)

    # Orphaned Compositions
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, mg_sizes, color="tab:purple", label="Total Meta-Genome Size", linestyle="--")
    ax.plot(gens, orphans_list, color="tab:red", label="Orphaned Compositions (0 instances)")
    mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("Count")
    ax.set_title(title("Orphaned Compositions (The Infinity Problem)"))
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer3_orphans.png"), dpi=150)
    plt.close(fig)

    # Boundary Source Attribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, base_bounds_list, color="tab:green", label="Baseline Insertions (Geological)")
    ax.plot(gens, open_bounds_list, color="tab:orange", label="Open Insertions (Representational)")
    mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("New Boundary Pairs per Generation")
    ax.set_title(title("Boundary Source Attribution"))
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer3_boundary_sources.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------
    # LAYER 0-2 PRESERVED PLOTS
    # -------------------------------------------------------------

    # Boundary geometry
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, A0_list, label="A0 (depth-0 tokens)")
    ax.plot(gens, A1_list, label="A1 (interior tokens)")
    ax.plot(gens, A2_list, label="A2 (boundary tokens)")
    mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("Token count (population total)")
    ax.set_title(title("Boundary Geometry Dynamics"))
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "boundary_geometry.png"), dpi=150)
    plt.close(fig)

    # Segment lengths
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(gens, mseg_list);   mark_singularity(axes[0])
    axes[0].set_title(title("Mean Segment Length")); axes[0].set_xlabel("Generation")
    axes[1].plot(gens, maxseg_list); mark_singularity(axes[1])
    axes[1].set_title(title("Max Segment Length"));  axes[1].set_xlabel("Generation")
    axes[2].plot(gens, segvar_list); mark_singularity(axes[2])
    axes[2].set_title(title("Segment Length Variance")); axes[2].set_xlabel("Generation")
    fig.suptitle(title("Segment Length Dynamics")); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "segment_lengths.png"), dpi=150)
    plt.close(fig)

    # Contact dynamics suite
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes[0, 0].plot(gens, fgap0_list, label="gap=0 (contact)")
    axes[0, 0].plot(gens, fgap1_list, label="gap=1 (expansion-ready)")
    axes[0, 0].plot(gens, fge2_list,  label="gap>=2 (free)")
    mark_singularity(axes[0, 0]); axes[0, 0].set_title(title("Gap State Distribution"))
    axes[0, 0].set_xlabel("Generation"); axes[0, 0].set_ylabel("Fraction of gaps"); axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(gens, fgap0_list, color="tab:red"); mark_singularity(axes[0, 1])
    axes[0, 1].set_title(title("Contact Rate (gap=0)")); axes[0, 1].set_xlabel("Generation")
    axes[0, 1].set_ylabel("Fraction of boundary pairs in contact")

    axes[0, 2].plot(gens, fgap1_list, color="tab:orange"); mark_singularity(axes[0, 2])
    axes[0, 2].set_title(title("Expansion-Ready Rate (gap=1)")); axes[0, 2].set_xlabel("Generation")
    axes[0, 2].set_ylabel("Fraction of boundary pairs at gap=1")

    axes[1, 0].plot(gens, flux_list, color="tab:purple", alpha=0.7); mark_singularity(axes[1, 0])
    axes[1, 0].set_title(title("Contact Flux (|Δ gap=0| per generation)")); axes[1, 0].set_xlabel("Generation")
    axes[1, 0].set_ylabel("Abs change in contact fraction")

    ax2 = axes[1, 1].twinx()
    axes[1, 1].plot(gens, mgap_list, color="tab:blue", label="mean gap")
    ax2.plot(gens, fgap0_list, color="tab:red", alpha=0.6, label="contact rate")
    mark_singularity(axes[1, 1]); axes[1, 1].set_title(title("Mean Gap vs Contact Rate"))
    axes[1, 1].set_xlabel("Generation"); axes[1, 1].set_ylabel("Mean gap length", color="tab:blue")
    ax2.set_ylabel("Contact fraction", color="tab:red")
    axes[1, 1].legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

    axes[1, 2].plot(gens, segvar_list, color="tab:green"); mark_singularity(axes[1, 2])
    axes[1, 2].set_title(title("Segment Length Variance")); axes[1, 2].set_xlabel("Generation")
    axes[1, 2].set_ylabel("Variance")

    fig.suptitle(title("Contact Dynamics Suite"), fontsize=14); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "contact_dynamics.png"), dpi=150)
    plt.close(fig)

    # Composition library growth
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, mg_sizes, color="tab:purple"); mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("Number of compositions")
    ax.set_title(title("Composition Library Growth")); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "composition_growth.png"), dpi=150)
    plt.close(fig)

    # Token type counts
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, base_counts,  label="Base tokens (2,3)",       color="tab:blue")
    ax.plot(gens, comp_counts,  label="Composition tokens (≥4)", color="tab:orange")
    ax.plot(gens, delim_counts, label="Delimiter tokens (0,1)",  color="tab:green", alpha=0.5)
    mark_singularity(ax); ax.set_xlabel("Generation"); ax.set_ylabel("Token count (population total)")
    ax.set_title(title("Token Type Distribution")); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "token_counts.png"), dpi=150)
    plt.close(fig)

    # Token fractions
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, base_fracs,    label="Base fraction",           color="tab:blue")
    ax.plot(gens, comp_fracs,    label="Composition fraction",    color="tab:orange")
    ax.plot(gens, dom_frac_list, label="Dominant comp fraction",  color="tab:red", linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 threshold")
    ax.axhline(SINGULARITY_THRESHOLD, color="darkred", linestyle="--", linewidth=0.8, label=f"Singularity threshold ({SINGULARITY_THRESHOLD:.0%})")
    mark_singularity(ax); ax.set_xlabel("Generation"); ax.set_ylabel("Fraction of non-delimiter tokens")
    ax.set_title(title("Base vs Composition Fractions + Dominant Composition"))
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "token_fractions.png"), dpi=150)
    plt.close(fig)

    # DAG dynamics suite
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes[0, 0].plot(gens, dag_nodes_list, color="tab:purple"); mark_singularity(axes[0, 0])
    axes[0, 0].set_title(title("DAG Nodes (= MG size)")); axes[0, 0].set_xlabel("Generation"); axes[0, 0].set_ylabel("Count")

    axes[0, 1].plot(gens, dag_edges_list, color="tab:blue"); mark_singularity(axes[0, 1])
    axes[0, 1].set_title(title("DAG Edges (composition references)")); axes[0, 1].set_xlabel("Generation"); axes[0, 1].set_ylabel("Count")

    axes[0, 2].plot(gens, dag_maxdepth_list,  color="tab:red",    label="max depth")
    axes[0, 2].plot(gens, dag_meandepth_list, color="tab:orange", label="mean depth", alpha=0.8)
    mark_singularity(axes[0, 2]); axes[0, 2].set_title(title("DAG Depth (compositional hierarchy)"))
    axes[0, 2].set_xlabel("Generation"); axes[0, 2].set_ylabel("Depth"); axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(gens, dag_roots_list, color="tab:green"); mark_singularity(axes[1, 0])
    axes[1, 0].set_title(title("DAG Roots (top-level compositions)")); axes[1, 0].set_xlabel("Generation")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].plot(gens, dag_lcc_list, color="tab:brown"); mark_singularity(axes[1, 1])
    axes[1, 1].set_title(title("Largest Connected Component")); axes[1, 1].set_xlabel("Generation"); axes[1, 1].set_ylabel("Nodes in LCC")

    edge_density = [dag_edges_list[i] / dag_nodes_list[i] if dag_nodes_list[i] > 0 else 0.0 for i in range(len(gens))]
    axes[1, 2].plot(gens, edge_density, color="tab:cyan"); mark_singularity(axes[1, 2])
    axes[1, 2].set_title(title("Edge Density (edges / nodes)")); axes[1, 2].set_xlabel("Generation"); axes[1, 2].set_ylabel("Avg out-degree")

    fig.suptitle(title("DAG Dynamics — Compositional Hierarchy"), fontsize=14); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "dag_dynamics.png"), dpi=150)
    plt.close(fig)

    print(f"\nDone. Outputs in ./{OUTPUT_FOLDER}/")

if __name__ == "__main__":
    run()