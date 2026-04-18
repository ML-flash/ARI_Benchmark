"""
theory_sim5.py

Layer 5: Metabolic Pipeline — PR, DB, Inlining, EA Recycling

Layer 0-4 (carried forward):
  Boundary geometry dynamics, MG, EA, DAG instrumentation,
  COMP_CHOICE_PROB, OPEN_PROB, MCO_DECAY all preserved exactly as
  implemented in Layer 4.

Layer 5 addition:
  The full metabolic pipeline activates. This is the first layer where
  the MG can shrink — compositions can die, identifiers get recycled,
  and the DAG undergoes structural pruning via inlining.

  New subsystems:
    - Participation Register (PR): A stack of capacity π₁₄.
      Each generation, the population is scanned for top-level
      composition tokens. Every composition found in the scan is moved
      to the top region of the stack (True-signal class). Compositions
      not found remain in their current position and drift toward the
      bottom (False-signal class). New captures are placed at the top.
      When the register is at capacity and a new composition must enter,
      the composition at the bottom (displacement boundary) is moved to
      the Deletion Basket.

    - Deletion Basket (DB): Displaced compositions accumulate age.
      Each generation, if a DB resident receives a participation signal
      (found in population scan), it is rescued back to the PR top with
      age reset to 0. If a DB resident reaches age π₁₅=2 without a
      participation signal, it dies.

    - Inlining: On death, every other composition in the MG that
      references the dying composition in its content has that reference
      replaced with the dying composition's expanded content. This
      preserves phenotypic identity. The dying composition is then
      removed from the MG entirely.

    - EA Recycling: The freed identifier goes to a reuse pool. New
      captures draw from the reuse pool first before incrementing the
      global counter.

    - MCO Modification: On death, the dying composition is removed from
      the MCO and the gap closes. On new capture (whether fresh or
      recycled ID), the new composition goes to MCO top.

  No fitness function. Participation signals come from scanning the
  population for top-level composition tokens — no decode required.

  Predicted dynamics:
    - MG size becomes non-monotonic for the first time.
    - The infinity/orphan problem should be actively managed.
    - DAG depth and MG size may exhibit cyclic build-destroy dynamics.
    - The singularity may be prevented, delayed, or transformed into
      a sustained oscillatory regime.

Outputs under ./theory_sim_5_ccp{}_op{}_mco{}_cap{}_del{}/:
  (All Layer 4 plots preserved)
  + layer5_metabolic_pipeline.png
  + layer5_mg_net_dynamics.png
  + layer5_register_state.png
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
GENERATIONS     = 5000
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
MUTATION_PROB           = 0.05
DELIMITED_MUTATION_PROB = 0.02

# Layer 3 Parameter: Outward Force
OPEN_PROB = 0.005

PROGRESS_EVERY = 20
SINGULARITY_THRESHOLD = 0.95

# Layer 2 Parameter
COMP_CHOICE_PROB = 0.43

# Layer 4 Parameter: MCO Decay (π₁₃)
MCO_DECAY = 0.90

# Layer 5 Parameters
PR_CAPACITY = 50       # π₁₄ — Participation Register capacity
DB_THRESHOLD = 2       # π₁₅ — Deletion Basket age threshold
MAX_INLINE_LEN = 80000   # Max content length after inlining; exceeding triggers cascade deletion

# Encoding constants
ENC_OPEN      = 0
ENC_CLOSE     = 1
ENC_BASE0     = 2
ENC_BASE1     = 3
FIRST_COMP_ID = 4

OUTPUT_FOLDER = (f"theory_sim_5_ccp{COMP_CHOICE_PROB:.2f}_op{OPEN_PROB:.2f}"
                 f"_mco{MCO_DECAY:.2f}_cap{PR_CAPACITY}_del{DB_THRESHOLD}")


# =========================
# META-GENOME + EA + MCO + PR + DB
# =========================

class MetaGenome:
    def __init__(self):
        self.store   = {}          # cid -> content_tuple
        self.reverse = {}          # content_tuple -> cid
        self.next_id = FIRST_COMP_ID
        self.deps    = {}          # cid -> set of cids it references
        self.rdeps   = {}          # cid -> set of cids that reference it

        # Layer 4: MCO stack — ordered list, oldest first, newest at end
        self.mco     = []

        # Layer 5: EA Reuse Pool
        self.reuse_pool = []

        # Layer 5: Participation Register — stack, bottom=index 0, top=end
        # Compositions at the top are most recently active.
        # Displacement occurs from bottom (index 0).
        self.pr = []

        # Layer 5: Deletion Basket — dict of cid -> age
        self.db = {}

        # Layer 5: Backflow detection — set True when inlining would overflow
        self.backflow_failure = False

    # ------ Layer 5: Participation Register Operations ------

    def pr_register(self, cid):
        """Add a newly captured composition to the PR top.
        If at capacity, displace from bottom to DB."""
        if cid in self.pr:
            return  # Already registered (shouldn't happen for new captures)
        if len(self.pr) >= PR_CAPACITY:
            self._pr_displace()
        self.pr.append(cid)

    def _pr_displace(self):
        """Move bottom of PR stack to Deletion Basket."""
        if not self.pr:
            return
        victim = self.pr.pop(0)
        self.db[victim] = 0  # Enter DB at age 0

    def pr_refresh(self, active_set):
        """Given the set of composition IDs with participation signals,
        move all True-signal compositions to the top region, preserving
        their relative ordering from the existing stack. A composition
        that was already higher stays higher within the True-signal group.
        False-signal compositions compact downward toward the displacement
        boundary, also preserving their relative ordering."""
        true_signal  = []
        false_signal = []
        for cid in self.pr:
            if cid in active_set:
                true_signal.append(cid)
            else:
                false_signal.append(cid)
        # False-signal at bottom (displacement boundary = index 0),
        # True-signal at top, both groups preserve original relative order.
        self.pr = false_signal + true_signal

    # ------ Layer 5: Deletion Basket Operations ------

    def db_tick(self, active_set):
        """Process one generation tick on the DB.
        - Rescue any DB resident that received a participation signal.
        - Age all remaining residents.
        - Kill any that reach the threshold.
        Returns (rescued_count, deleted_ids)."""
        rescued = []
        aged    = {}
        dead    = []

        for cid, age in self.db.items():
            if cid in active_set:
                rescued.append(cid)
            else:
                new_age = age + 1
                if new_age >= DB_THRESHOLD:
                    dead.append(cid)
                else:
                    aged[cid] = new_age

        # Process rescues — back to PR top
        for cid in rescued:
            del self.db[cid]  # Remove from DB first
            if len(self.pr) >= PR_CAPACITY:
                self._pr_displace()
            self.pr.append(cid)

        # Process deaths — sort by content length (shortest first) to
        # minimize inlining cascade.  Inlining a short composition adds
        # little content to its parents. Processing short deaths first
        # prevents large cascading expansions.
        dead.sort(key=lambda cid: len(self.store.get(cid, ())))
        deleted_ids = []
        for cid in dead:
            if cid in self.db:
                del self.db[cid]
            self._kill_composition(cid)
            if self.backflow_failure:
                break
            deleted_ids.append(cid)

        # Update ages for survivors
        for cid, age in aged.items():
            self.db[cid] = age

        return len(rescued), deleted_ids

    # ------ Layer 5: Inlining + Death ------

    def _kill_composition(self, cid):
        """Inline and remove a composition from the system.
        If inlining would make a parent's content exceed MAX_INLINE_LEN,
        sets self.backflow_failure = True and aborts."""
        if cid not in self.store:
            return

        content = self.store[cid]

        # Inline: replace references to cid in all other compositions
        referencing = list(self.rdeps.get(cid, set()))
        for ref_cid in referencing:
            if ref_cid not in self.store:
                continue
            old_content = self.store[ref_cid]

            # Check for backflow — inlining would exceed content cap
            ref_count = old_content.count(cid)
            new_len = len(old_content) - ref_count + ref_count * len(content)

            if new_len > MAX_INLINE_LEN:
                self.backflow_failure = True
                return

            # Expand: replace every occurrence of cid with content
            new_content = []
            for tok in old_content:
                if tok == cid:
                    new_content.extend(content)
                else:
                    new_content.append(tok)
            new_tuple = tuple(new_content)

            # Update reverse mapping
            if old_content in self.reverse:
                del self.reverse[old_content]
            self.store[ref_cid] = new_tuple
            self.reverse[new_tuple] = ref_cid

            # Update deps for ref_cid: remove cid, add cid's deps
            self.deps[ref_cid].discard(cid)
            cid_deps = self.deps.get(cid, set())
            self.deps[ref_cid].update(cid_deps)
            # Update rdeps for cid's deps: they now have ref_cid as parent
            for dep in cid_deps:
                if dep in self.rdeps:
                    self.rdeps[dep].add(ref_cid)

        # Remove cid from rdeps of its dependencies
        for dep in self.deps.get(cid, set()):
            if dep in self.rdeps:
                self.rdeps[dep].discard(cid)

        # Remove cid from all data structures
        content_key = self.store[cid]
        if content_key in self.reverse:
            del self.reverse[content_key]
        del self.store[cid]
        if cid in self.deps:
            del self.deps[cid]
        if cid in self.rdeps:
            del self.rdeps[cid]

        # Remove from MCO
        if cid in self.mco:
            self.mco.remove(cid)

        # Remove from PR (shouldn't be there — compositions enter _kill
        # only from DB death — but guard against inconsistency)
        if cid in self.pr:
            self.pr.remove(cid)

        # Recycle ID
        self.reuse_pool.append(cid)

    # ------ Capture (modified for EA recycling) ------

    def try_capture(self, content_tuple):
        if len(content_tuple) < MIN_CAPTURE_LEN:
            return None, False
        if content_tuple in self.reverse:
            return self.reverse[content_tuple], False
        # Allocate ID: reuse pool first, then fresh
        if self.reuse_pool:
            cid = self.reuse_pool.pop()
        else:
            cid = self.next_id
            self.next_id += 1

        self.store[cid]             = content_tuple
        self.reverse[content_tuple] = cid

        referenced = set(tok for tok in content_tuple if tok >= FIRST_COMP_ID)
        self.deps[cid] = referenced
        self.rdeps.setdefault(cid, set())
        for ref in referenced:
            self.rdeps.setdefault(ref, set()).add(cid)

        # MCO: append to top
        self.mco.append(cid)

        # PR: register new composition
        self.pr_register(cid)

        return cid, True

    # ------ MCO Selection (unchanged from Layer 4) ------

    def size(self):
        return len(self.store)

    def all_ids(self):
        return list(self.store.keys())

    def sample_composition(self, rng):
        """Sample a composition ID from the MCO stack using exponential
        decay. p(mⱼ) ∝ π₁₃^(|M| - j - 1)."""
        n = len(self.mco)
        if n == 0:
            return None

        weights = []
        for j in range(n):
            weights.append(MCO_DECAY ** (n - j - 1))

        total = sum(weights)
        roll = rng.random() * total
        cumulative = 0.0
        for j in range(n):
            cumulative += weights[j]
            if roll <= cumulative:
                return self.mco[j]
        return self.mco[-1]

    def mco_entropy(self):
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
        n = len(self.mco)
        if n == 0:
            return 0
        weights_newest_first = [MCO_DECAY ** k for k in range(n)]
        total = sum(weights_newest_first)
        cumulative = 0.0
        for count, w in enumerate(weights_newest_first, 1):
            cumulative += w / total
            if cumulative >= 0.90:
                return count
        return n

    def mco_rank_of(self, cid):
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

        edges = sum(len(self.deps.get(i, set())) for i in ids)
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
        if cid is not None:
            org[open_idx : close_idx + 1] = [cid]
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
                if len(org) - 2 >= MIN_LEN:
                    i = remove_pair_at(org, i)
                    inside = rescan_inside(org, i) if org else False
                    continue
                else:
                    i += 1
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
            # Check composition still exists in MG (Layer 5: it may have been deleted)
            if tok not in mg.store:
                # Dead reference — treat as inert, skip open
                i += 1
                continue
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
# LAYER 5: POPULATION SCAN FOR PARTICIPATION
# =========================

def scan_participation(pop):
    """Scan all organisms for top-level composition tokens.
    Returns the set of composition IDs found (the service set)."""
    active = set()
    for org in pop:
        for tok in org:
            if tok >= FIRST_COMP_ID:
                active.add(tok)
    return active


def run_metabolic_pipeline(mg, active_set, events):
    """Execute one generation of the metabolic pipeline.
    1. Refresh PR with participation signals.
    2. Tick DB (rescue/age/kill).
    Returns (rescued_count, deleted_ids)."""
    # Step 1: Refresh PR — move participating compositions to top
    mg.pr_refresh(active_set)

    # Step 2: Tick DB — rescue, age, kill
    rescued, deleted_ids = mg.db_tick(active_set)

    events["rescues"]   += rescued
    events["deletions"] += len(deleted_ids)

    return rescued, deleted_ids


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

    # Orphan tracking
    orphans = sum(1 for cid in mg.all_ids() if cid not in comp_freq)
    orphan_frac = orphans / mg_size if mg_size > 0 else 0.0

    # MCO metrics
    mco_ent = mg.mco_entropy()
    mco_win = mg.mco_effective_window()
    mco_dom_rank = mg.mco_rank_of(dom_id) if dom_id >= FIRST_COMP_ID else -1

    top5_ids = set(mg.mco[-5:]) if len(mg.mco) >= 5 else set(mg.mco)
    top5_count = sum(comp_freq.get(cid, 0) for cid in top5_ids)
    mco_top5_frac = top5_count / non_delim if non_delim else 0.0

    # Layer 5 metrics
    pr_occupancy = len(mg.pr)
    db_occupancy = len(mg.db)
    reuse_pool_size = len(mg.reuse_pool)

    return (
        delim_count, base_count, comp_count, base_frac, comp_frac, mg_size, alpha_size,
        A0, A1, A2,
        num_pairs, mean_seg, seg_max, seg_var,
        num_gaps, mean_gap, min_gap, frac_le_1,
        frac_gap_0, frac_gap_1, frac_ge_2,
        dom_id, dom_frac,
        orphans, orphan_frac,
        mco_ent, mco_win, mco_dom_rank, mco_top5_frac,
        pr_occupancy, db_occupancy, reuse_pool_size
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

    param_label = (f"ccp={COMP_CHOICE_PROB:.2f} op={OPEN_PROB:.2f} "
                   f"mco={MCO_DECAY:.2f} cap={PR_CAPACITY} del={DB_THRESHOLD}")

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

    # Layer 5 Lists
    pr_occupancy_list    = []
    db_occupancy_list    = []
    reuse_pool_list      = []
    rescues_list         = []
    deletions_list       = []
    cumulative_deletions = 0
    cumulative_deletions_list = []

    singularity_gen       = None
    comp_exceeds_base_gen = None
    prev_fgap0            = None

    events = {"captures": 0, "opens": 0, "baseline_bounds": 0, "open_bounds": 0,
              "rescues": 0, "deletions": 0}

    csv_path = os.path.join(OUTPUT_FOLDER, "composition_growth_layer5.csv")
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
            "mco_dominant_rank", "mco_top5_pop_frac",
            "pr_occupancy", "db_occupancy", "reuse_pool_size",
            "rescues", "deletions", "cumulative_deletions"
        ])

        t0 = time.time()

        for gen in range(GENERATIONS + 1):

            # --- Layer 5: Metabolic pipeline runs each generation ---
            backflow_gen = None
            if gen > 0:
                active_set = scan_participation(pop)
                rescued, deleted_ids = run_metabolic_pipeline(mg, active_set, events)
                cumulative_deletions += len(deleted_ids)

                if mg.backflow_failure:
                    backflow_gen = gen
                    print(f"\n  *** BACKFLOW FAILURE at generation {gen} ***")
                    print(f"      Inlining cascade exceeded MAX_INLINE_LEN={MAX_INLINE_LEN}")
                    print(f"      Entropy exhaust blocked — no fitness function to dissipate.")
                    print(f"      Recording final state and generating plots.\n")

            (delim_count, base_count, comp_count, base_frac, comp_frac, mg_size, alpha_size,
             A0, A1, A2,
             num_pairs, mean_seg, seg_max, seg_var,
             num_gaps, mean_gap, min_gap, frac_le_1,
             frac_gap_0, frac_gap_1, frac_ge_2,
             dom_id, dom_frac,
             orphans, orphan_frac,
             mco_ent, mco_win, mco_dom_rank, mco_top5_frac,
             pr_occ, db_occ, reuse_sz) = compute_stats(pop, mg)

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
                mco_dom_rank, f"{mco_top5_frac:.6f}",
                pr_occ, db_occ, reuse_sz,
                events["rescues"], events["deletions"], cumulative_deletions
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

            captures_list.append(events["captures"])
            opens_list.append(events["opens"])
            base_bounds_list.append(events["baseline_bounds"])
            open_bounds_list.append(events["open_bounds"])
            orphans_list.append(orphans)
            orphan_frac_list.append(orphan_frac)

            mco_entropy_list.append(mco_ent)
            mco_window_list.append(mco_win)
            mco_dom_rank_list.append(mco_dom_rank)
            mco_top5_frac_list.append(mco_top5_frac)

            pr_occupancy_list.append(pr_occ)
            db_occupancy_list.append(db_occ)
            reuse_pool_list.append(reuse_sz)
            rescues_list.append(events["rescues"])
            deletions_list.append(events["deletions"])
            cumulative_deletions_list.append(cumulative_deletions)

            if gen % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  gen {gen:4d}  mg={mg_size:5d}  "
                      f"base={base_frac:.3f}  comp={comp_frac:.3f}  "
                      f"orphans={orphan_frac:.2%}  dag_depth={dag_md:3d}  "
                      f"mco_win={mco_win:4d}  top5={mco_top5_frac:.3f}  "
                      f"pr={pr_occ:3d}  db={db_occ:3d}  "
                      f"del={events['deletions']:3d}  res={events['rescues']:3d}  "
                      f"reuse={reuse_sz:5d}  "
                      f"elapsed={elapsed:.1f}s")

            # Reset events
            events = {"captures": 0, "opens": 0, "baseline_bounds": 0, "open_bounds": 0,
                      "rescues": 0, "deletions": 0}

            if backflow_gen is not None:
                break

            if gen < GENERATIONS:
                pop = make_next_generation(pop, rng)
                mutate_population(pop, mg, rng, events)

    if singularity_gen is not None:
        print(f"\n  *** Singularity at generation {singularity_gen} ***")
    if backflow_gen is not None:
        print(f"  *** Backflow failure at generation {backflow_gen} ***")
    if singularity_gen is None and backflow_gen is None:
        print(f"\n  Singularity not reached within {GENERATIONS} generations.")

    # ================================================================
    # PLOTS
    # ================================================================

    sg  = singularity_gen
    sg2 = comp_exceeds_base_gen
    bf  = backflow_gen

    def mark_singularity(ax):
        if sg2 is not None:
            ax.axvline(sg2, color="orange", linestyle="--", linewidth=0.9, label=f"Comps > base (gen {sg2})")
        if sg is not None:
            ax.axvline(sg, color="red", linestyle=":", linewidth=1.2, label=f"Singularity (gen {sg})")
        if bf is not None:
            ax.axvline(bf, color="black", linestyle="-", linewidth=2.0, alpha=0.7, label=f"BACKFLOW (gen {bf})")

    def title(base):
        return f"{base}  [{param_label}]"

    # -------------------------------------------------------------
    # LAYER 5 PLOTS
    # -------------------------------------------------------------

    # 1. Metabolic Pipeline Activity (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(gens, captures_list, color="tab:green", alpha=0.8, label="Captures (births)")
    axes[0].plot(gens, deletions_list, color="tab:red", alpha=0.8, label="Deletions (deaths)")
    axes[0].plot(gens, rescues_list, color="tab:blue", alpha=0.7, label="Rescues")
    mark_singularity(axes[0])
    axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Events per Generation")
    axes[0].set_title(title("Metabolic Activity"))
    axes[0].legend(fontsize=8)

    axes[1].plot(gens, pr_occupancy_list, color="tab:purple", label=f"PR Occupancy (cap={PR_CAPACITY})")
    axes[1].plot(gens, db_occupancy_list, color="tab:orange", label="DB Occupancy")
    axes[1].axhline(PR_CAPACITY, color="purple", linestyle="--", linewidth=0.8, alpha=0.5)
    mark_singularity(axes[1])
    axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Count")
    axes[1].set_title(title("Register / Basket Occupancy"))
    axes[1].legend(fontsize=8)

    axes[2].plot(gens, reuse_pool_list, color="tab:cyan", label="EA Reuse Pool Size")
    axes[2].plot(gens, cumulative_deletions_list, color="tab:red", linestyle="--",
                 alpha=0.6, label="Cumulative Deletions")
    mark_singularity(axes[2])
    axes[2].set_xlabel("Generation"); axes[2].set_ylabel("Count")
    axes[2].set_title(title("Identifier Recycling"))
    axes[2].legend(fontsize=8)

    fig.suptitle(title("Layer 5: Metabolic Pipeline"), fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer5_metabolic_pipeline.png"), dpi=150)
    plt.close(fig)

    # 2. MG Net Dynamics — first time MG can shrink
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(gens, mg_sizes, color="tab:purple", linewidth=1.5, label="MG Size (net)")
    ax1.plot(gens, orphans_list, color="tab:red", alpha=0.6, label="Orphaned Compositions")
    ax2.plot(gens, cumulative_deletions_list, color="tab:gray", linestyle="--",
             alpha=0.7, label="Cumulative Deletions")
    mark_singularity(ax1)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Count", color="tab:purple")
    ax2.set_ylabel("Cumulative Deletions", color="tab:gray")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle(title("MG Net Dynamics (Non-Monotonic)"), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer5_mg_net_dynamics.png"), dpi=150)
    plt.close(fig)

    # 3. Register State Over Time
    fig, ax = plt.subplots(figsize=(10, 5))
    total_tracked = [pr_occupancy_list[i] + db_occupancy_list[i] for i in range(len(gens))]
    ax.fill_between(gens, 0, pr_occupancy_list, color="tab:purple", alpha=0.4, label="PR")
    ax.fill_between(gens, pr_occupancy_list, total_tracked, color="tab:orange", alpha=0.4, label="DB")
    ax.plot(gens, mg_sizes, color="black", linewidth=1.0, linestyle="--", label="Total MG Size")
    ax.axhline(PR_CAPACITY, color="purple", linestyle=":", linewidth=0.8, alpha=0.6,
               label=f"PR Capacity ({PR_CAPACITY})")
    mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("Count")
    ax.set_title(title("Register State: PR + DB vs Total MG"))
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer5_register_state.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------
    # LAYER 4 PRESERVED PLOTS
    # -------------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(gens, mco_entropy_list, color="tab:blue")
    mark_singularity(axes[0])
    axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Shannon Entropy (bits)")
    axes[0].set_title(title("MCO Selection Entropy"))

    axes[1].plot(gens, mco_window_list, color="tab:green", label="Effective Window (90%)")
    axes[1].plot(gens, mg_sizes, color="tab:purple", linestyle="--", alpha=0.6, label="MG Size")
    mark_singularity(axes[1])
    axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Count")
    axes[1].set_title(title("Effective Selection Window vs MG Size"))
    axes[1].legend(fontsize=8)

    axes[2].plot(gens, mco_top5_frac_list, color="tab:orange")
    mark_singularity(axes[2])
    axes[2].set_xlabel("Generation"); axes[2].set_ylabel("Fraction of non-delimiter tokens")
    axes[2].set_title(title("Top-5 MCO Population Fraction"))

    fig.suptitle(title("MCO Selection Dynamics"), fontsize=14); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer4_mco_selection_dynamics.png"), dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(gens, dom_frac_list, color="tab:red", linewidth=1.5, label="Dominant Comp Fraction")
    ax1.axhline(SINGULARITY_THRESHOLD, color="darkred", linestyle="--", linewidth=0.8,
                label=f"Singularity ({SINGULARITY_THRESHOLD:.0%})")
    ax1.set_xlabel("Generation"); ax1.set_ylabel("Dominant Comp Fraction", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2.plot(gens, mco_dom_rank_list, color="tab:blue", alpha=0.7, label="Dominant Comp MCO Rank")
    ax2.set_ylabel("MCO Rank (0 = newest)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    mark_singularity(ax1)
    ax1.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle(title("MCO Rank vs Singularity Formation"), fontsize=13); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer4_mco_vs_singularity.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------
    # LAYER 3 PRESERVED PLOTS
    # -------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, captures_list, color="tab:red", alpha=0.8, label="Successful Captures (Inward Force)")
    ax.plot(gens, opens_list, color="tab:blue", alpha=0.8, label="Successful Opens (Outward Force)")
    mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("Events per Generation")
    ax.set_title(title("Thermodynamic Pressure (Capture vs. Open Flux)"))
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer3_pressure_flux.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, mg_sizes, color="tab:purple", label="Total Meta-Genome Size", linestyle="--")
    ax.plot(gens, orphans_list, color="tab:red", label="Orphaned Compositions (0 instances)")
    mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("Count")
    ax.set_title(title("Orphaned Compositions (The Infinity Problem)"))
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "layer3_orphans.png"), dpi=150)
    plt.close(fig)

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

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, mg_sizes, color="tab:purple"); mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("Number of compositions")
    ax.set_title(title("Composition Library Growth")); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "composition_growth.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, base_counts,  label="Base tokens (2,3)",       color="tab:blue")
    ax.plot(gens, comp_counts,  label="Composition tokens (≥4)", color="tab:orange")
    ax.plot(gens, delim_counts, label="Delimiter tokens (0,1)",  color="tab:green", alpha=0.5)
    mark_singularity(ax); ax.set_xlabel("Generation"); ax.set_ylabel("Token count (population total)")
    ax.set_title(title("Token Type Distribution")); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "token_counts.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, base_fracs,    label="Base fraction",           color="tab:blue")
    ax.plot(gens, comp_fracs,    label="Composition fraction",    color="tab:orange")
    ax.plot(gens, dom_frac_list, label="Dominant comp fraction",  color="tab:red", linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 threshold")
    ax.axhline(SINGULARITY_THRESHOLD, color="darkred", linestyle="--", linewidth=0.8,
               label=f"Singularity threshold ({SINGULARITY_THRESHOLD:.0%})")
    mark_singularity(ax); ax.set_xlabel("Generation"); ax.set_ylabel("Fraction of non-delimiter tokens")
    ax.set_title(title("Base vs Composition Fractions + Dominant Composition"))
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "token_fractions.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, alpha_sizes, color="teal"); mark_singularity(ax)
    ax.set_xlabel("Generation"); ax.set_ylabel("Alphabet size (4 + MG size)")
    ax.set_title(title("Alphabet Size")); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, "alphabet_size.png"), dpi=150)
    plt.close(fig)

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