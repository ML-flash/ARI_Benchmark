import os
import json
import math
import random
import copy
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ----------------------------------------------------------------------------
# DEPENDENCIES
# ----------------------------------------------------------------------------
try:
    from M_E_GA import M_E_GA_Base
except ImportError:
    print("Error: Could not import M_E_GA_Base from M_E_GA.")
    exit()

try:
    from TSS_Benchmark_Large import TSS_Benchmark, apply_chemistry_fast, score_sack, PROPS
except ImportError:
    print("Error: Could not import from TSS_Benchmark_Large.")
    exit()

# ----------------------------------------------------------------------------
# GLOBAL PARAMETERS
# ----------------------------------------------------------------------------
GLOBAL_SEED = None  # MEGA evolutionary seed — None for stochastic runs
ENV_SEED = 42069    # Environment seed — fixed standard candle matching benchmark
random.seed(GLOBAL_SEED)

VOLUME = 40
NUM_ITEMS = 7000
NUM_GROUPS = 6
MAX_SIZE = 45
MAX_WEIGHT = 35
MAX_DENSITY = 35

CELL_SIZE = 5

best_organism = {"genome": None, "fitness": float("-inf")}


def update_best_organism(current_genome, current_fitness, verbose=True):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness:.4f}")


# ----------------------------------------------------------------------------
# SPATIAL ENTROPY
# ----------------------------------------------------------------------------

def compute_spatial_entropy(items, volume, cell_size):
    grid_span = 2 * volume + 1
    num_cells_per_axis = max(1, grid_span // cell_size)

    def cell_index(pos):
        x, y, z = pos
        cx = ((x + volume) // cell_size) % num_cells_per_axis
        cy = ((y + volume) // cell_size) % num_cells_per_axis
        cz = ((z + volume) // cell_size) % num_cells_per_axis
        return (cx, cy, cz)

    cell_counts = Counter()
    positioned = 0
    for item in items:
        if item['position'] is not None:
            positioned += 1
            cell_counts[cell_index(item['position'])] += 1

    if positioned == 0:
        return 0.0, 0.0

    total = float(positioned)
    entropy = 0.0
    for count in cell_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    total_cells = num_cells_per_axis ** 3
    max_entropy = math.log2(min(positioned, total_cells))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return entropy, norm_entropy


# ----------------------------------------------------------------------------
# ROLLING WINDOW HELPER
# ----------------------------------------------------------------------------

def _rolling_best(values: list, window: int) -> list:
    """Max over a sliding window. Identical logic applied to both signals."""
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        result.append(max(values[start_idx: i + 1]))
    return result


# ----------------------------------------------------------------------------
# ARI ANALYSIS
# ----------------------------------------------------------------------------

def compute_ari(gen_data):
    """
    Attractor Resolution Index (ARI).

    Measures whether the agent systematically restructures the environment
    in ways that produce better outcomes over time.

    ARI = rho(rolling_entropy_reduction, rolling_best_fitness) * (max_cum_reduction / H0)

    Both signals use the same rolling window for temporal consistency.
    Displacement remains cumulative as the global agency signal.
    """
    gens = sorted(gen_data.keys())
    if len(gens) < 3:
        return {'ari': 0.0}

    entropies = [gen_data[g]['norm_entropy'] for g in gens]
    max_fits = [gen_data[g]['max_fitness'] for g in gens]
    mean_fits = [gen_data[g]['mean_fitness'] for g in gens]

    # =====================================================================
    # SYMMETRIC ROLLING WINDOW
    # =====================================================================
    h0 = entropies[0]
    entropy_reductions = [h0 - entropies[i] for i in range(len(gens))]

    rolling_window = 5
    rolling_entropy_reduction = _rolling_best(entropy_reductions, rolling_window)
    rolling_best_fitness = _rolling_best(max_fits, rolling_window)

    rho_cum, p_rho_cum = scipy_stats.spearmanr(rolling_entropy_reduction, rolling_best_fitness)
    r_cum, p_cum = scipy_stats.pearsonr(rolling_entropy_reduction, rolling_best_fitness)

    # =====================================================================
    # ARI: MULTIPLICATIVE FORMULATION
    # =====================================================================
    max_cum_reduction = max(entropy_reductions)

    if h0 > 0 and max_cum_reduction > 0:
        entropy_displacement = max_cum_reduction / h0
    else:
        entropy_displacement = 0.0

    ari = rho_cum * entropy_displacement

    # =====================================================================
    # DELTA ANALYSIS (diagnostic only)
    # =====================================================================
    entropy_deltas = [entropies[i] - entropies[i-1] for i in range(1, len(gens))]
    max_fit_deltas = [max_fits[i] - max_fits[i-1] for i in range(1, len(gens))]
    delta_gens = gens[1:]

    r_delta, p_delta = scipy_stats.pearsonr(entropy_deltas, max_fit_deltas)
    rho_delta, p_rho_delta = scipy_stats.spearmanr(entropy_deltas, max_fit_deltas)
    ari_delta = -r_delta

    concordant = sum(1 for i in range(len(entropy_deltas))
                     if entropy_deltas[i] < 0 and max_fit_deltas[i] > 0)
    discordant = sum(1 for i in range(len(entropy_deltas))
                     if entropy_deltas[i] > 0 and max_fit_deltas[i] > 0)
    total_entropy_drops = sum(1 for d in entropy_deltas if d < 0)
    total_fitness_gains = sum(1 for d in max_fit_deltas if d > 0)

    # Entropy trend
    entropy_slope, entropy_intercept = np.polyfit(
        np.array(gens, dtype=float), np.array(entropies, dtype=float), 1
    )
    total_entropy_change = entropies[-1] - entropies[0]
    entropy_change_ratio = entropies[-1] / entropies[0] if entropies[0] > 0 else 1.0

    # =====================================================================
    # REPORTING
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"Attractor Resolution Index (ARI)")
    print(f"{'='*60}")

    print(f"\n  Entropy trajectory:")
    print(f"    Initial:  {entropies[0]:.6f}")
    print(f"    Final:    {entropies[-1]:.6f}")
    print(f"    Change:   {total_entropy_change:.6f}")
    print(f"    Ratio:    {entropy_change_ratio:.6f}")
    print(f"    Slope:    {entropy_slope:.8f} per gen")

    print(f"\n  Fitness trajectory:")
    print(f"    Initial max:              {max_fits[0]:.2f}")
    print(f"    Final max:                {max_fits[-1]:.2f}")
    print(f"    Overall best:             {max(max_fits):.2f}")
    print(f"    Rolling best (final):     {rolling_best_fitness[-1]:.2f}")

    print(f"\n  --- ARI (multiplicative) ---")
    print(f"    Spearman rho (rolling, window={rolling_window}): {rho_cum:.4f}")
    print(f"    p-value:                   {p_rho_cum:.6f}")
    print(f"    Max cum entropy reduction: {max_cum_reduction:.6f}")
    print(f"    Entropy displacement:      {entropy_displacement:.6f}")
    print(f"    ARI = rho * displacement:  {ari:.6f}")

    print(f"\n  Rolling entropy reduction range: "
          f"{min(rolling_entropy_reduction):.6f} to {max(rolling_entropy_reduction):.6f}")
    print(f"  Rolling best fitness range: "
          f"{min(rolling_best_fitness):.2f} to {max(rolling_best_fitness):.2f}")

    print(f"\n  --- DELTA CORRELATIONS (diagnostic) ---")
    print(f"    Pearson  r={r_delta:.4f}  p={p_delta:.6f}")
    print(f"    Spearman rho={rho_delta:.4f}  p={p_rho_delta:.6f}")
    print(f"    ARI_delta = {ari_delta:.4f}")
    print(f"    Concordant: {concordant}  Discordant: {discordant}")

    print(f"\n{'='*60}")

    return {
        # Primary metric
        'ari': ari,
        'entropy_displacement': entropy_displacement,
        'max_cum_reduction': max_cum_reduction,

        # Rolling window components
        'rho_cum': rho_cum, 'p_rho_cum': p_rho_cum,
        'r_cum': r_cum, 'p_cum': p_cum,
        'rolling_entropy_reduction': rolling_entropy_reduction,
        'rolling_best_fitness': rolling_best_fitness,
        'entropy_reductions': entropy_reductions,

        # Delta (diagnostic)
        'ari_delta': ari_delta,
        'r_delta': r_delta, 'p_delta': p_delta,
        'rho_delta': rho_delta, 'p_rho_delta': p_rho_delta,

        # Concordance
        'concordant': concordant, 'discordant': discordant,
        'total_entropy_drops': total_entropy_drops,
        'total_fitness_gains': total_fitness_gains,

        # Trajectory data
        'entropy_slope': entropy_slope,
        'entropy_intercept': entropy_intercept,
        'entropy_change_ratio': entropy_change_ratio,
        'total_entropy_change': total_entropy_change,
        'gens': gens,
        'entropies': entropies,
        'max_fits': max_fits,
        'mean_fits': mean_fits,
        'entropy_deltas': entropy_deltas,
        'max_fit_deltas': max_fit_deltas,
        'delta_gens': delta_gens,
    }


# ----------------------------------------------------------------------------
# PANEL DIMENSIONS
# ----------------------------------------------------------------------------

PANEL_DPI = 150
PANEL_WIDTH_PX = 560
PANEL_HEIGHT_PX = 280
PANEL_FIGSIZE = (PANEL_WIDTH_PX / PANEL_DPI, PANEL_HEIGHT_PX / PANEL_DPI)
PANEL_DIR = "panels"


def _ensure_panel_dir():
    os.makedirs(PANEL_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# INDIVIDUAL PANEL GENERATORS (560x280)
# ----------------------------------------------------------------------------

def _panel_raw_trajectories(results):
    path = os.path.join(PANEL_DIR, "panel_raw_trajectories.png")
    gens = results['gens']
    fig, ax1 = plt.subplots(figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    ax1.set_ylabel('Norm Entropy', color='tab:blue', fontsize=8)
    ax1.plot(gens, results['entropies'], '-', color='tab:blue', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=7)
    ax1.tick_params(axis='x', labelsize=7)
    ax1r = ax1.twinx()
    ax1r.set_ylabel('Max Fitness', color='tab:red', fontsize=8)
    ax1r.plot(gens, results['max_fits'], '-', color='tab:red', linewidth=1, alpha=0.7)
    ax1r.tick_params(axis='y', labelcolor='tab:red', labelsize=7)
    ax1.set_title('Raw Trajectories', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Generation', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=PANEL_DPI)
    plt.close(fig)
    print(f"  Panel saved: {path}")


def _panel_rolling_window(results):
    path = os.path.join(PANEL_DIR, "panel_rolling_window.png")
    gens = results['gens']
    roll_h = results['rolling_entropy_reduction']
    roll_f = results['rolling_best_fitness']
    fig, ax2 = plt.subplots(figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    ax2.set_ylabel('Rolling Entropy Reduction', color='tab:blue', fontsize=8)
    ax2.plot(gens, roll_h, '-', color='tab:blue', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=7)
    ax2.tick_params(axis='x', labelsize=7)
    ax2r = ax2.twinx()
    ax2r.set_ylabel('Rolling Best Fitness', color='tab:red', fontsize=8)
    ax2r.plot(gens, roll_f, '-', color='tab:red', linewidth=2)
    ax2r.tick_params(axis='y', labelcolor='tab:red', labelsize=7)
    ax2.set_title(f'Symmetric Rolling Window (\u03c1={results["rho_cum"]:.3f})', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Generation', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=PANEL_DPI)
    plt.close(fig)
    print(f"  Panel saved: {path}")


def _panel_delta_scatter(results):
    path = os.path.join(PANEL_DIR, "panel_delta_scatter.png")
    e_deltas = results['entropy_deltas']
    f_deltas = results['max_fit_deltas']
    fig, ax3 = plt.subplots(figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    ax3.scatter(e_deltas, f_deltas, alpha=0.5, s=15, c='tab:purple')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Entropy Delta', fontsize=8)
    ax3.set_ylabel('Max Fitness Delta', fontsize=8)
    ax3.set_title(f'Delta Scatter (r={results["r_delta"]:.3f})', fontsize=9, fontweight='bold')
    ax3.tick_params(labelsize=7)
    xlim = ax3.get_xlim()
    ylim = ax3.get_ylim()
    ax3.text(xlim[0]*0.7, ylim[1]*0.8, 'Construction', fontsize=7, color='green', ha='center')
    ax3.text(xlim[1]*0.7, ylim[0]*0.8, 'Destruction', fontsize=7, color='red', ha='center')
    plt.tight_layout()
    plt.savefig(path, dpi=PANEL_DPI)
    plt.close(fig)
    print(f"  Panel saved: {path}")


def _panel_rolling_scatter(results):
    path = os.path.join(PANEL_DIR, "panel_rolling_scatter.png")
    gens = results['gens']
    roll_h = results['rolling_entropy_reduction']
    roll_f = results['rolling_best_fitness']
    fig, ax4 = plt.subplots(figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    scatter = ax4.scatter(roll_h, roll_f, c=gens, cmap='viridis', s=20, alpha=0.7)
    ax4.set_xlabel('Rolling Entropy Reduction', fontsize=8)
    ax4.set_ylabel('Rolling Best Fitness', fontsize=8)
    ax4.set_title(f'Rolling Scatter (\u03c1={results["rho_cum"]:.3f})', fontsize=9, fontweight='bold')
    ax4.tick_params(labelsize=7)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Generation', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=PANEL_DPI)
    plt.close(fig)
    print(f"  Panel saved: {path}")


def _panel_rolling_delta_correlation(results):
    path = os.path.join(PANEL_DIR, "panel_rolling_delta_corr.png")
    e_deltas = results['entropy_deltas']
    f_deltas = results['max_fit_deltas']
    delta_gens = results['delta_gens']
    window = 10
    fig, ax5 = plt.subplots(figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    if len(e_deltas) >= window:
        rolling_r = []
        rolling_gens = []
        for i in range(len(e_deltas) - window + 1):
            try:
                r, _ = scipy_stats.pearsonr(e_deltas[i:i+window], f_deltas[i:i+window])
                rolling_r.append(-r)
            except:
                rolling_r.append(0.0)
            rolling_gens.append(delta_gens[i + window // 2])
        ax5.plot(rolling_gens, rolling_r, 'g-', linewidth=1.5)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.fill_between(rolling_gens, rolling_r, 0, alpha=0.2,
                         where=[r > 0 for r in rolling_r], color='green')
        ax5.fill_between(rolling_gens, rolling_r, 0, alpha=0.2,
                         where=[r <= 0 for r in rolling_r], color='red')
    ax5.set_xlabel('Generation', fontsize=8)
    ax5.set_ylabel('Rolling ARI (window=10)', fontsize=8)
    ax5.set_title('Rolling Delta Correlation', fontsize=9, fontweight='bold')
    ax5.set_ylim(-1.1, 1.1)
    ax5.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=PANEL_DPI)
    plt.close(fig)
    print(f"  Panel saved: {path}")


def _panel_concordance(results):
    path = os.path.join(PANEL_DIR, "panel_concordance.png")
    e_deltas = results['entropy_deltas']
    f_deltas = results['max_fit_deltas']
    h_down_f_up = results['concordant']
    h_up_f_up = results['discordant']
    h_down_f_down = sum(1 for i in range(len(e_deltas)) if e_deltas[i] < 0 and f_deltas[i] < 0)
    h_up_f_down = sum(1 for i in range(len(e_deltas)) if e_deltas[i] > 0 and f_deltas[i] < 0)
    labels = ['Construction\n(H\u2193 F\u2191)', 'Degradation\n(H\u2191 F\u2191)',
              'Waste\n(H\u2193 F\u2193)', 'Destruction\n(H\u2191 F\u2193)']
    counts = [h_down_f_up, h_up_f_up, h_down_f_down, h_up_f_down]
    colors = ['green', 'orange', 'gray', 'red']
    fig, ax6 = plt.subplots(figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    bars = ax6.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    for bar, count in zip(bars, counts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha='center', fontsize=8)
    ax6.set_ylabel('Count', fontsize=8)
    ax6.set_title('Generation Event Classification', fontsize=9, fontweight='bold')
    ax6.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=PANEL_DPI)
    plt.close(fig)
    print(f"  Panel saved: {path}")


# ----------------------------------------------------------------------------
# PLOTTING (composite + individual panels)
# ----------------------------------------------------------------------------

def plot_analysis(results, plot_path="ARI_analysis.png", fitness_plot_path="fitness_over_time.png"):
    """Six-panel ARI analysis plot + dedicated Max Fitness line chart + individual panels."""
    gens = results['gens']
    entropies = results['entropies']
    max_fits = results['max_fits']
    delta_gens = results['delta_gens']
    e_deltas = results['entropy_deltas']
    f_deltas = results['max_fit_deltas']
    roll_h = results['rolling_entropy_reduction']
    roll_f = results['rolling_best_fitness']

    # ====================================================================
    # 6-PANEL COMPOSITE PLOT
    # ====================================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(
        f"ARI Analysis  |  ARI={results['ari']:.6f}  "
        f"(rho={results['rho_cum']:.4f} \u00d7 disp={results['entropy_displacement']:.6f})  "
        f"|  Delta={results['ari_delta']:.4f} (p={results['p_delta']:.4f})",
        fontsize=13
    )

    # Panel 1: Raw trajectories
    ax1 = axes[0, 0]
    ax1.set_ylabel('Norm Entropy', color='tab:blue')
    ax1.plot(gens, entropies, '-', color='tab:blue', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1r = ax1.twinx()
    ax1r.set_ylabel('Max Fitness', color='tab:red')
    ax1r.plot(gens, max_fits, '-', color='tab:red', linewidth=1, alpha=0.7)
    ax1r.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title('Raw Trajectories')
    ax1.set_xlabel('Generation')

    # Panel 2: Symmetric rolling trajectories
    ax2 = axes[0, 1]
    ax2.set_ylabel('Rolling Entropy Reduction', color='tab:blue')
    ax2.plot(gens, roll_h, '-', color='tab:blue', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2r = ax2.twinx()
    ax2r.set_ylabel('Rolling Best Fitness', color='tab:red')
    ax2r.plot(gens, roll_f, '-', color='tab:red', linewidth=2)
    ax2r.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_title(f'Symmetric Rolling Window (rho={results["rho_cum"]:.3f})')
    ax2.set_xlabel('Generation')

    # Panel 3: Delta scatter
    ax3 = axes[1, 0]
    ax3.scatter(e_deltas, f_deltas, alpha=0.5, s=20, c='tab:purple')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Entropy Delta')
    ax3.set_ylabel('Max Fitness Delta')
    ax3.set_title(f'Delta Scatter (r={results["r_delta"]:.3f})')
    xlim = ax3.get_xlim(); ylim = ax3.get_ylim()
    ax3.text(xlim[0]*0.7, ylim[1]*0.8, 'Construction', fontsize=8, color='green', ha='center')
    ax3.text(xlim[1]*0.7, ylim[0]*0.8, 'Destruction', fontsize=8, color='red', ha='center')

    # Panel 4: Symmetric rolling scatter
    ax4 = axes[1, 1]
    scatter = ax4.scatter(roll_h, roll_f, c=gens, cmap='viridis', s=30, alpha=0.7)
    ax4.set_xlabel('Rolling Entropy Reduction')
    ax4.set_ylabel('Rolling Best Fitness')
    ax4.set_title(f'Symmetric Rolling Scatter (rho={results["rho_cum"]:.3f})')
    plt.colorbar(scatter, ax=ax4, label='Generation')

    # Panel 5: Rolling delta correlation
    ax5 = axes[2, 0]
    window = 10
    if len(e_deltas) >= window:
        rolling_r = []
        rolling_gens = []
        for i in range(len(e_deltas) - window + 1):
            try:
                r, _ = scipy_stats.pearsonr(e_deltas[i:i+window], f_deltas[i:i+window])
                rolling_r.append(-r)
            except:
                rolling_r.append(0.0)
            rolling_gens.append(delta_gens[i + window // 2])
        ax5.plot(rolling_gens, rolling_r, 'g-', linewidth=1.5)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.fill_between(rolling_gens, rolling_r, 0, alpha=0.2,
                         where=[r > 0 for r in rolling_r], color='green')
        ax5.fill_between(rolling_gens, rolling_r, 0, alpha=0.2,
                         where=[r <= 0 for r in rolling_r], color='red')
    ax5.set_xlabel('Generation')
    ax5.set_ylabel('Rolling ARI (window=10)')
    ax5.set_title('Rolling Delta Correlation')
    ax5.set_ylim(-1.1, 1.1)

    # Panel 6: Concordance summary
    ax6 = axes[2, 1]
    labels = ['Construction\n(H\u2193 F\u2191)', 'Degradation\n(H\u2191 F\u2191)',
              'Waste\n(H\u2193 F\u2193)', 'Destruction\n(H\u2191 F\u2193)']
    h_down_f_up = results['concordant']
    h_up_f_up = results['discordant']
    h_down_f_down = sum(1 for i in range(len(e_deltas)) if e_deltas[i] < 0 and f_deltas[i] < 0)
    h_up_f_down = sum(1 for i in range(len(e_deltas)) if e_deltas[i] > 0 and f_deltas[i] < 0)
    counts = [h_down_f_up, h_up_f_up, h_down_f_down, h_up_f_down]
    colors = ['green', 'orange', 'gray', 'red']
    bars = ax6.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    for bar, count in zip(bars, counts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', fontsize=10)
    ax6.set_ylabel('Count')
    ax6.set_title('Generation Event Classification')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_path, dpi=300)
    print(f"Composite analysis plot saved: {plot_path}")
    plt.close(fig)

    # ====================================================================
    # DEDICATED MAX FITNESS PLOT
    # ====================================================================
    fig2, ax_fit = plt.subplots(figsize=(10, 5))
    ax_fit.plot(gens, max_fits, color='tab:red', linewidth=2, label='Max Fitness')

    peak_val = max(max_fits)
    peak_idx = max_fits.index(peak_val)
    peak_gen = gens[peak_idx]

    ax_fit.annotate(f'Peak: {peak_val:.0f}',
                xy=(peak_gen, peak_val),
                xytext=(peak_gen + 5, peak_val),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold')

    ax_fit.set_title("Max Fitness Trajectory Over Generations", fontsize=14, pad=15)
    ax_fit.set_xlabel("Generation", fontsize=12)
    ax_fit.set_ylabel("Max Fitness", fontsize=12)
    ax_fit.grid(True, linestyle='--', alpha=0.6)
    ax_fit.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fitness_plot_path, dpi=300)
    print(f"Fitness plot saved: {fitness_plot_path}")
    plt.close(fig2)

    # ====================================================================
    # INDIVIDUAL PANELS (560x280)
    # ====================================================================
    _ensure_panel_dir()
    print(f"\nGenerating individual panels ({PANEL_WIDTH_PX}x{PANEL_HEIGHT_PX}) in {PANEL_DIR}/...")
    _panel_raw_trajectories(results)
    _panel_rolling_window(results)
    _panel_delta_scatter(results)
    _panel_rolling_scatter(results)
    _panel_rolling_delta_correlation(results)
    _panel_concordance(results)
    print(f"All panels saved to {PANEL_DIR}/")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def run_experiment():
    print("Starting TSS \u2014 Attractor Resolution Index Benchmark (MEGA)")
    print("=" * 60)

    fitness_function = TSS_Benchmark(
        volume=VOLUME, num_items=NUM_ITEMS, num_groups=NUM_GROUPS,
        update_best_func=update_best_organism,
        max_size=MAX_SIZE, max_weight=MAX_WEIGHT, max_density=MAX_DENSITY,
        seed=ENV_SEED,
    )

    config = {
        "mutation_prob": 0.02,
        "delimited_mutation_prob": 0.01,
        "open_mutation_prob": 0.0011,
        "metagene_mutation_prob": 0.0015,
        "delimiter_insert_prob": 0.002,
        "delimit_delete_prob": 0.008,
        "crossover_prob": 0.00,
        "elitism_ratio": 0.00,
        "base_gene_prob": 0.90,     # π12
        "metagene_prob": 0.50,      # π13
        "max_individual_length": 100,
        "population_size": 500,
        "num_parents": 200,
        "max_generations": 2000,
        "delimiters": False,
        "delimiter_space": 2,
        "logging": False,
        "generation_logging": False,
        "mutation_logging": False,
        "crossover_logging": False,
        "individual_logging": False,
        "seed": GLOBAL_SEED,
        "lru_cache_size": 45,
    }

    ga = M_E_GA_Base(
        fitness_function.genes,
        lambda ind, ga_instance: fitness_function.compute(ind, ga_instance),
        **config,
    )

    gen_data = {}

    def before_generation_finalize(ga_instance):
        gen = getattr(ga_instance, "current_generation", 0)
        raw_h, norm_h = compute_spatial_entropy(
            fitness_function.current_items, VOLUME, CELL_SIZE
        )
        fits = getattr(ga_instance, "fitness_scores", [])
        if fits:
            best_fit = float(np.max(fits))
            mean_fit = float(np.mean(fits))
            fitness_function.flush_generation()
            gen_data[gen] = {
                'raw_entropy': raw_h,
                'norm_entropy': norm_h,
                'max_fitness': best_fit,
                'mean_fitness': mean_fit,
            }
            print(f"  Gen {gen:4d} | best={best_fit:.2f} mean={mean_fit:.2f} | entropy={norm_h:.6f}")

    ga.before_generation_finalize = before_generation_finalize
    ga.run_algorithm()

    best = best_organism
    decoded = ga.decode_organism(best["genome"], format=True)

    print(f"\nExperiment Complete")
    print(f"Best Fitness:              {best['fitness']:.4f}")
    print(f"Cumulative Adjusted Delta: {fitness_function.get_cumulative_delta():.4f}")
    print(f"Length of Best Solution:   {len(decoded)}")
    print(f"Generations tracked:       {len(gen_data)}")

    results = compute_ari(gen_data)
    plot_analysis(results)

    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"  Best Fitness:             {best['fitness']:.4f}")
    print(f"  ARI:                      {results['ari']:.6f}")
    print(f"    Spearman rho (rolling, window=5): {results['rho_cum']:.4f}")
    print(f"    Entropy displacement:   {results['entropy_displacement']:.6f}")
    print(f"    Max cum reduction:      {results['max_cum_reduction']:.6f}")
    print(f"    p-value (rolling):      {results['p_rho_cum']:.6f}")
    print(f"  ARI_delta (diagnostic):   {results['ari_delta']:.4f}")
    print(f"    Pearson r:              {results['r_delta']:.4f}")
    print(f"    p-value:                {results['p_delta']:.6f}")
    print(f"  Entropy change ratio:     {results['entropy_change_ratio']:.6f}")
    print(f"  Concordant/Discordant:    {results['concordant']}/{results['discordant']}")


if __name__ == "__main__":
    run_experiment()