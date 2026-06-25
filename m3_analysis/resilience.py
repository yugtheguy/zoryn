#!/usr/bin/env python3
"""
Module 10 — Resilience Index
Computes macro network robustness index comparing baseline vs damaged topologies.
Implements the formula: ResilienceIndex = BaselineAverageShortestPath / DamagedAverageShortestPath.
Assigns qualitative vulnerability classifications (Highly Resilient, Moderately Vulnerable, High Vulnerability).
"""

import os
import argparse
import json

def interpret_resilience_score(score: float) -> str:
    """Assigns qualitative assessment based on ResilienceIndex."""
    if score >= 0.85:
        return "Highly resilient"
    elif score >= 0.50:
        return "Moderately vulnerable"
    else:
        return "High vulnerability"

def compute_resilience_index(simulation_results: dict) -> dict:
    """
    Evaluates network resilience index across stress testing simulation scenarios.
    """
    baseline = simulation_results.get("baseline", {})
    base_asp = baseline.get("lcc_average_shortest_path", 0.0)
    base_eff = baseline.get("global_efficiency", 0.0)
    
    multi_node = simulation_results.get("multi_node_failures", {})
    
    scenarios_eval = {}
    
    for sc_name, sc_data in multi_node.items():
        k = sc_data.get("k", 1)
        metrics_after = sc_data.get("metrics_after_failure", {})
        damaged_asp = metrics_after.get("lcc_average_shortest_path", 0.0)
        damaged_eff = metrics_after.get("global_efficiency", 0.0)
        
        # ASP Resilience Index
        if damaged_asp > 0 and base_asp > 0:
            # Note: damaged ASP usually increases, so base/damaged <= 1.0
            asp_idx = round(min(1.0, base_asp / damaged_asp), 4)
        elif damaged_asp == 0 and base_asp == 0:
            asp_idx = 1.0
        else:
            asp_idx = 0.0
            
        # Efficiency Resilience Index (supplementary metric)
        if base_eff > 0:
            eff_idx = round(min(1.0, damaged_eff / base_eff), 4)
        else:
            eff_idx = 0.0
            
        # Composite resilience index
        comp_res = round((asp_idx + eff_idx) / 2.0 if base_asp > 0 else eff_idx, 4)
        
        scenarios_eval[sc_name] = {
            "k_knockouts": k,
            "baseline_asp": base_asp,
            "damaged_asp": damaged_asp,
            "asp_resilience_index": asp_idx,
            "efficiency_resilience_index": eff_idx,
            "composite_resilience_index": comp_res,
            "interpretation": interpret_resilience_score(comp_res)
        }
        
    # Overall benchmark score (e.g., top-5 or top-3 failure scenario)
    benchmark_sc = scenarios_eval.get("top_5_knockouts") or next(iter(scenarios_eval.values()), {})
    overall_idx = benchmark_sc.get("composite_resilience_index", 1.0)
    
    report = {
        "overall_resilience_index": overall_idx,
        "overall_classification": interpret_resilience_score(overall_idx),
        "baseline_metrics": {
            "average_shortest_path": base_asp,
            "global_efficiency": base_eff
        },
        "scenarios": scenarios_eval
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="M3 Module 10: Resilience Index")
    parser.add_argument('--input_sim', type=str, required=True, help="Input simulation_results JSON")
    parser.add_argument('--output_resilience', type=str, default="resilience_report.json", help="Output resilience JSON")
    args = parser.parse_args()
    
    with open(args.input_sim, 'r') as f:
        sim_data = json.load(f)
        
    res_report = compute_resilience_index(sim_data)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_resilience)), exist_ok=True)
    with open(args.output_resilience, 'w') as f:
        json.dump(res_report, f, indent=2)
    print(f"Resilience report saved to: {os.path.abspath(args.output_resilience)}")

if __name__ == "__main__":
    main()
