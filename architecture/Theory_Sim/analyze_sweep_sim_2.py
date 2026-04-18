import os
import csv
import glob

def analyze_sweep():
    # Find all the output folders from the sweep
    folders = glob.glob("theory_sim_2_ccp*")
    
    if not folders:
        print("No folders found matching 'theory_sim_2_ccp*'")
        return

    print(f"{'CCP':<6} | {'Singularity Gen':<17} | {'Max MG Size':<12} | {'A0 Variance':<12} | {'Boom-Bust Cycles (A0)':<25}")
    print("-" * 80)

    results = []

    for folder in folders:
        ccp_val = float(folder.split("ccp")[-1])
        csv_path = os.path.join(folder, "composition_growth.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            gens = []
            a0_vals = []
            mg_sizes = []
            dom_fracs = []
            
            for row in reader:
                gens.append(int(row["gen"]))
                a0_vals.append(int(row["A0"]))
                mg_sizes.append(int(row["mg_size"]))
                dom_fracs.append(float(row["dominant_comp_frac"]))
                
        # Find singularity (when dominant comp hits 1.0)
        singularity_gen = "Did Not Reach"
        for g, df in zip(gens, dom_fracs):
            if df >= 1.0:
                singularity_gen = str(g)
                break
                
        max_mg = max(mg_sizes) if mg_sizes else 0
        
        # Calculate A0 variance to detect the "wild cycling"
        mean_a0 = sum(a0_vals) / len(a0_vals) if a0_vals else 0
        var_a0 = sum((x - mean_a0)**2 for x in a0_vals) / len(a0_vals) if a0_vals else 0
        
        # Count "Boom-Bust" cycles by looking for major direction changes in A0
        peaks = 0
        for i in range(1, len(a0_vals) - 1):
            if a0_vals[i] > a0_vals[i-1] and a0_vals[i] > a0_vals[i+1] and a0_vals[i] > mean_a0:
                peaks += 1
                
        results.append((ccp_val, singularity_gen, max_mg, int(var_a0), peaks))

    # Sort by CCP value
    results.sort(key=lambda x: x[0])

    for r in results:
        print(f"{r[0]:<6.2f} | {r[1]:<17} | {r[2]:<12} | {r[3]:<12} | {r[4]:<25}")

if __name__ == "__main__":
    analyze_sweep()