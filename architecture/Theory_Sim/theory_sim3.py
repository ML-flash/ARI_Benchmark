"""
theory_sim3.py

Layer 3: The Outward Force (Open Mutation)

Layer 0 & 1 & 2 (carried forward):
  Boundary geometry dynamics, MG, EA, DAG instrumentation, and
  COMP_CHOICE_PROB all preserved exactly as implemented in Layer 2.

Layer 3 addition:
  OPEN_PROB controls the outward energetic force. When a point mutation
  targets a composition token, it may unpack it into its constituent parts.
  
  Context-Dependent Unpacking (A11.2 Structural Integrity):
    - Outside boundary: Unpacks and wraps in [Start, End] to preserve
      the structural substrate and generate a new delimited region.
    - Inside boundary: Unpacks content directly without adding new
      delimiters to prevent violating the no-nesting invariant.

  New Telemetry:
    - Capture vs. Open Flux: Tracks the thermodynamic pressure of the system.
    - Orphaned Compositions: Tracks the accumulation of compositions with
      zero active instances in the population (The Infinity Problem).
    - Boundary Source Attribution: Delineates between baseline geological
      insertion and representation-driven Open insertion.

Outputs under ./theory_sim_3_ccp{COMP_CHOICE_PROB}_op{OPEN_PROB}/:
  (All Layer 2 plots)
  + layer3_pressure_flux.png
  + layer3_orphans.png
  + layer3_boundary_sources.png
"""

import os
import csv
import random
import time
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================

POPULATION_SIZE = 500
GENERATIONS     = 5000
SEED            = 1

MIN_LEN = 2
MAX_LEN = 100

ENABLE_CROSSOVER = True
NUM_PARENTS      = 200
CROSSOVER_PROB   = 0.50

# Boundary operators
BOUNDARY_INSERT_PROB = 0.0035
BOUNDARY_REMOVE_PROB = 0.0017

# Capture probability
CAPTURE_PROB = 0.09
MIN_CAPTURE_LEN = 2

# Basic mutation rates
MUTATION_PROB           = 0.05
DELIMITED_MUTATION_PROB = 0.02

# Layer 3 Parameter: Outward Force
OPEN_PROB = 0.005

PROGRESS_EVERY = 20
SINGULARITY_THRESHOLD = 0.95

# Layer 2 Parameter
COMP_CHOICE_PROB = 0.55

# Encoding constants
ENC_OPEN      = 0
ENC_CLOSE     = 1
ENC_BASE0     = 2
ENC_BASE1     = 3
FIRST_COMP_ID = 4

OUTPUT_FOLDER = f"theory_sim_3_ccp{COMP_CHOICE_PROB:.2f}_op{OPEN_PROB:.2f}"


# =========================
# META-GENOME + EA
# =========================

class MetaGenome:
    def __init__(self):
        self.store   = {}
        self.reverse = {}
        self.next_id = FIRST_COMP_ID
        self.deps    = {}
        self.rdeps   = {}

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

        return cid, True

    def size(self):
        return len(self.store)

    def all_ids(self):
        return list(self.store.keys())

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
    comp_ids = mg.all_ids()
    if comp_ids and rng.random() < COMP_CHOICE_PROB:
        return rng.choice(comp_ids)
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

    # -- Layer 3: Orphan Tracking --
    orphans = sum(1 for cid in mg.all_ids() if cid not in comp_freq)
    orphan_frac = orphans / mg_size if mg_size > 0 else 0.0

    return (
        delim_count, base_count, comp_count, base_frac, comp_frac, mg_size, alpha_size,
        A0, A1, A2,
        num_pairs, mean_seg, seg_max, seg_var,
        num_gaps, mean_gap, min_gap, frac_le_1,
        frac_gap_0, frac_gap_1, frac_ge_2,
        dom_id, dom_frac,
        orphans, orphan_frac
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

    param_label = f"ccp={COMP_CHOICE_PROB:.2f} op={OPEN_PROB:.2f}"

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

    singularity_gen       = None
    comp_exceeds_base_gen = None
    prev_fgap0            = None
    
    events = {"captures": 0, "opens": 0, "baseline_bounds": 0, "open_bounds": 0}

    csv_path = os.path.join(OUTPUT_FOLDER, "composition_growth_layer3.csv")
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
            "orphans", "orphan_frac"
        ])

        t0 = time.time()

        for gen in range(GENERATIONS + 1):

            (delim_count, base_count, comp_count, base_frac, comp_frac, mg_size, alpha_size,
             A0, A1, A2,
             num_pairs, mean_seg, seg_max, seg_var,
             num_gaps, mean_gap, min_gap, frac_le_1,
             frac_gap_0, frac_gap_1, frac_ge_2,
             dom_id, dom_frac,
             orphans, orphan_frac) = compute_stats(pop, mg)

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
                orphans, f"{orphan_frac:.6f}"
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
            
            # Layer 3 telemetry appends
            captures_list.append(events["captures"])
            opens_list.append(events["opens"])
            base_bounds_list.append(events["baseline_bounds"])
            open_bounds_list.append(events["open_bounds"])
            orphans_list.append(orphans)
            orphan_frac_list.append(orphan_frac)

            if gen % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  gen {gen:4d}  mg={mg_size:5d}  "
                      f"base={base_frac:.3f}  comp={comp_frac:.3f}  "
                      f"orphans={orphan_frac:.2%}  dag_lcc={dag_lcc:4d}  "
                      f"cap/open={events['captures']}/{events['opens']}  "
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
    # NEW LAYER 3 PLOTS
    # -------------------------------------------------------------
    
    # 1. Capture vs Open Flux (Thermodynamic Pressure)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, captures_list, color="tab:red", alpha=0.8, label="Successful Captures (Inward Force)")
    ax.plot(gens, opens_list, color="tab:blue", alpha=0.8, label="Successful Opens (Outward Force)")
    mark_singularity(ax)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Events per Generation")
    ax.set_title(title("Thermodynamic Pressure (Capture vs. Open Flux)"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer3_pressure_flux.png"), dpi=150)
    plt.close(fig)

    # 2. Orphaned Compositions (The Infinity Problem)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, mg_sizes, color="tab:purple", label="Total Meta-Genome Size", linestyle="--")
    ax.plot(gens, orphans_list, color="tab:red", label="Orphaned Compositions (0 instances)")
    mark_singularity(ax)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.set_title(title("Orphaned Compositions (The Infinity Problem)"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer3_orphans.png"), dpi=150)
    plt.close(fig)

    # 3. Boundary Source Attribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, base_bounds_list, color="tab:green", label="Baseline Insertions (Geological)")
    ax.plot(gens, open_bounds_list, color="tab:orange", label="Open Insertions (Representational)")
    mark_singularity(ax)
    ax.set_xlabel("Generation")
    ax.set_ylabel("New Boundary Pairs per Generation")
    ax.set_title(title("Boundary Source Attribution"))
    ax.legend()
    fig.tight_layout()
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