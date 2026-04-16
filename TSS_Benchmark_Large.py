import random
import json
import os
import datetime
import uuid
import copy


PROPS = ('size', 'weight', 'density', 'value')


def apply_chemistry_fast(sack):
    """
    Apply bidirectional chemistry in pickup order.
    Builds property arrays without mutating original items.
    Returns list of dicts {size, weight, density, value}.
    """
    prop_arrays = []
    for item in sack:
        props = {p: item['properties'][p] for p in PROPS}
        prop_arrays.append(props)
    for k in range(len(sack)):
        item_k = sack[k]
        for j in range(k):
            item_j = sack[j]
            for interact in item_k.get('interactions', []):
                if item_j['group'] == interact['target_group']:
                    mag = interact['magnitude'] * item_k['reaction_strength']
                    change = mag if interact['direction'] == 'increase' else -mag
                    prop_arrays[j][interact['property']] = max(
                        0.1, prop_arrays[j][interact['property']] + change)
            for interact in item_j.get('interactions', []):
                if item_k['group'] == interact['target_group']:
                    mag = interact['magnitude'] * item_j['reaction_strength']
                    change = mag if interact['direction'] == 'increase' else -mag
                    prop_arrays[k][interact['property']] = max(
                        0.1, prop_arrays[k][interact['property']] + change)
    return prop_arrays


def score_sack(sack, max_size, max_weight, max_density):
    """
    Post-hoc constraint enforcement with cascading removal.
    10-point penalty per removed item. Full chemistry recalc on each removal.
    Returns (value minus penalties, item count).
    """
    working = list(sack)
    overpack_penalty = 0.0

    while working:
        pa = apply_chemistry_fast(working)
        if (sum(p['size'] for p in pa) <= max_size
                and sum(p['weight'] for p in pa) <= max_weight
                and sum(p['density'] for p in pa) <= max_density):
            return sum(p['value'] for p in pa) - overpack_penalty, len(pa)
        working.pop()
        overpack_penalty += 10.0

    return -overpack_penalty, 0


class TSS_Benchmark:
    """
    TSS fitness function aligned with the Kaggle benchmark physics.
    """

    def __init__(self, volume, num_items, num_groups, update_best_func,
                 max_size, max_weight, max_density, log_enabled=False,
                 seed=42069):
        self.update_best = update_best_func
        self.volume = volume
        self.num_items = num_items
        self.num_groups = num_groups
        self.max_size = max_size
        self.max_weight = max_weight
        self.max_density = max_density
        self.log_enabled = log_enabled

        self.genes = ['R', 'L', 'U', 'D', 'F', 'B', 'DR']
        self.directions = {
            'R': (1, 0, 0), 'L': (-1, 0, 0),
            'U': (0, 1, 0), 'D': (0, -1, 0),
            'F': (0, 0, 1), 'B': (0, 0, -1)
        }

        self.step_reward = 2.0
        self.step_penalty = -10.0
        self.soft_step_limit = 200
        self.drop_reward = 0.001
        self.value_multiplier_base = 1.75

        rng = random.Random(seed)
        self.current_items = self._create_items(rng)

        # Frozen at initialization. Never mutated.
        self.t0_items = copy.deepcopy(self.current_items)

        # Final resting position of last evaluated individual.
        self.last_best_pos = (0, 0, 0)

        self.log_filename = (
            f"logs/mega_tse_trajectory_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        if self.log_enabled:
            self._ensure_log_directory()

        # Drop telemetry — reset each generation via flush_generation
        self._current_dr_in_paths   = 0
        self._current_dr_attempted  = 0
        self._current_dr_succeeded  = 0
        self._current_items_picked  = 0

    def _ensure_log_directory(self):
        log_dir = os.path.dirname(self.log_filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(self.log_filename, 'w') as f:
            init_data = {
                "type": "initial_state",
                "config": {
                    "volume": self.volume,
                    "soft_step_limit": self.soft_step_limit
                },
                "agent_start_pos": [0, 0, 0],
                "items": copy.deepcopy(self.current_items)
            }
            f.write(json.dumps(init_data) + '\n')

    def _create_items(self, rng):
        all_positions = [
            (x, y, z)
            for x in range(-self.volume, self.volume + 1)
            for y in range(-self.volume, self.volume + 1)
            for z in range(-self.volume, self.volume + 1)
        ]
        rng.shuffle(all_positions)
        items = []
        for item_id in range(self.num_items):
            props = {
                'size':    rng.uniform(1, 30),
                'weight':  rng.uniform(1, 47),
                'density': rng.uniform(1, 50),
                'value':   rng.uniform(0.1, 2000.0),
            }
            item = {
                'id': item_id,
                'group': rng.randint(0, self.num_groups - 1),
                'properties': props,
                'base_properties': copy.deepcopy(props),
                'reaction_strength': rng.uniform(0.01, 50.0),
                'interactions': [],
                'position': all_positions[item_id],
            }
            num_interactions = rng.randint(0, self.num_groups - 1)
            other_groups = [g for g in range(self.num_groups) if g != item['group']]
            if num_interactions > 0 and other_groups:
                targets = rng.sample(other_groups, min(num_interactions, len(other_groups)))
                for tg in targets:
                    item['interactions'].append({
                        'target_group': tg,
                        'property':    rng.choice(['size', 'weight', 'density', 'value']),
                        'direction':   rng.choice(['increase', 'decrease']),
                        'magnitude':   rng.uniform(0.01, 10.0),
                    })
            items.append(item)
        return items

    def _wrap(self, c):
        v = self.volume
        return ((c + v) % (2 * v + 1)) - v

    def _evaluate_path(self, decoded_path):
        x, y, z = 0, 0, 0
        fitness = 0.0
        step = 0
        sack = []
        pickup_locations = {}
        handled = set()

        dr_in_path  = decoded_path.count('DR')
        dr_attempted  = 0
        dr_succeeded  = 0
        items_picked  = 0

        for gene in decoded_path:
            if gene == 'DR':
                if sack:
                    dr_attempted += 1
                    occupied = any(
                        i['position'] == (x, y, z) for i in self.current_items
                    )
                    if not occupied:
                        item = sack.pop(0)
                        item['position'] = (x, y, z)
                        item['properties'] = copy.deepcopy(item['base_properties'])
                        if item['id'] in pickup_locations:
                            del pickup_locations[item['id']]
                        handled.discard(item['id'])
                        fitness += self.drop_reward
                        dr_succeeded += 1

            elif gene in self.directions:
                dx, dy, dz = self.directions[gene]
                x = self._wrap(x + dx)
                y = self._wrap(y + dy)
                z = self._wrap(z + dz)
                fitness += self.step_reward if step < self.soft_step_limit else self.step_penalty
                step += 1

                at_pos = next(
                    (i for i in self.current_items
                     if i['position'] == (x, y, z) and i['id'] not in handled),
                    None,
                )
                if at_pos:
                    pickup_locations[at_pos['id']] = (x, y, z)
                    at_pos['position'] = None
                    handled.add(at_pos['id'])
                    sack.append(at_pos)
                    items_picked += 1

        if step > 0 and sack:
            sack_value, item_count = score_sack(
                sack, self.max_size, self.max_weight, self.max_density
            )
            fitness += sack_value
            fitness *= self.value_multiplier_base ** item_count

        for item in sack:
            if item['id'] in pickup_locations:
                item['position'] = pickup_locations[item['id']]
                item['properties'] = copy.deepcopy(item['base_properties'])

        self.last_best_pos = (x, y, z)

        # Accumulate into generation-level counters
        self._current_dr_in_paths  += dr_in_path
        self._current_dr_attempted += dr_attempted
        self._current_dr_succeeded += dr_succeeded
        self._current_items_picked += items_picked

        return fitness, step, sack, (x, y, z)

    def _step_contribution(self, step_count):
        rewarded  = min(step_count, self.soft_step_limit)
        penalized = max(0, step_count - self.soft_step_limit)
        return (rewarded * self.step_reward) + (penalized * self.step_penalty)

    def compute(self, encoded_individual, ga_instance):
        decoded = ga_instance.decode_organism(encoded_individual, format=True)
        raw_fitness, step_count, sack, final_pos = self._evaluate_path(decoded)
        adjusted = raw_fitness - self._step_contribution(step_count)

        if not hasattr(self, '_gen_raw_sums'):
            self._gen_raw_sums      = []
            self._gen_adjusted_sums = []
            self._current_raw       = 0.0
            self._current_adjusted  = 0.0
            self._current_count     = 0

        self._current_raw      += raw_fitness
        self._current_adjusted += adjusted
        self._current_count    += 1

        if self.log_enabled:
            self._log_evaluation(decoded, sack, raw_fitness, adjusted, final_pos, step_count)

        self.update_best(encoded_individual, raw_fitness)
        return raw_fitness

    def flush_generation(self):
        if not hasattr(self, '_gen_raw_sums'):
            self._gen_raw_sums      = []
            self._gen_adjusted_sums = []
            self._current_raw       = 0.0
            self._current_adjusted  = 0.0
            self._current_count     = 0

        raw_sum      = self._current_raw
        adjusted_sum = self._current_adjusted
        count        = self._current_count

        self._gen_raw_sums.append(raw_sum)
        self._gen_adjusted_sums.append(adjusted_sum)

        # Capture and print drop telemetry before reset
        dr_in   = self._current_dr_in_paths
        dr_att  = self._current_dr_attempted
        dr_succ = self._current_dr_succeeded
        picked  = self._current_items_picked

        print(f"  [drops] DR_in_paths={dr_in} attempted={dr_att} "
              f"succeeded={dr_succ} items_picked={picked}")

        # Reset all counters
        self._current_raw          = 0.0
        self._current_adjusted     = 0.0
        self._current_count        = 0
        self._current_dr_in_paths  = 0
        self._current_dr_attempted = 0
        self._current_dr_succeeded = 0
        self._current_items_picked = 0

        return raw_sum, adjusted_sum, count

    def get_generation_deltas(self):
        if not hasattr(self, '_gen_adjusted_sums') or not self._gen_adjusted_sums:
            return []
        sums = self._gen_adjusted_sums
        deltas = [sums[0]] + [sums[i] - sums[i-1] for i in range(1, len(sums))]
        return deltas

    def get_cumulative_delta(self):
        return sum(self.get_generation_deltas())

    def get_ari_states(self):
        return self.t0_items, self.current_items, self.last_best_pos

    def _log_evaluation(self, decoded, sack, raw_fitness, adjusted_fitness,
                        final_position, step_count):
        if not self.log_enabled:
            return
        log_entry = {
            "type": "evaluation",
            "evaluation_id": str(uuid.uuid4()),
            "path": decoded,
            "sack_summary": {
                "num_items":   len(sack),
                "items":       [i['id'] for i in sack],
                "total_value": sum(i['properties']['value'] for i in sack),
            },
            "raw_fitness":      raw_fitness,
            "adjusted_fitness": adjusted_fitness,
            "step_count":       step_count,
            "final_position":   final_position,
        }
        with open(self.log_filename, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')