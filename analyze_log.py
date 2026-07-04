"""
Generates charts + a research-style markdown report from a simulation_v2.py log file.

Usage:
    python analyze_log.py simulation_log_v2.json --out report/
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(path):
    with open(path) as f:
        return json.load(f)


def make_charts(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gens = [d["generation"] for d in data]

    plt.style.use("dark_background")
    palette = {"swarm": "#9b59b6", "swarm_avg": "#c39bd3", "pred": "#e74c3c", "pred_avg": "#f1948a"}

    # 1. Arms race: best fitness of each species over time
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, [d["swarm"]["best_fitness"] for d in data], color=palette["swarm"], label="Swarm best fitness", linewidth=2)
    ax.plot(gens, [d["swarm"]["avg_fitness"] for d in data], color=palette["swarm_avg"], linestyle="--", label="Swarm avg fitness", linewidth=1)
    ax.plot(gens, [d["predator"]["best_fitness"] for d in data], color=palette["pred"], label="Predator best fitness", linewidth=2)
    ax.plot(gens, [d["predator"]["avg_fitness"] for d in data], color=palette["pred_avg"], linestyle="--", label="Predator avg fitness", linewidth=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Co-Evolutionary Arms Race: Swarm vs Predator Fitness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_arms_race_fitness.png"), dpi=130)
    plt.close(fig)

    # 2. Network complexity growth (NEAT-lite hidden layer size)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gens, [d["swarm"]["avg_hidden_complexity"] for d in data], color=palette["swarm"], label="Swarm avg hidden neurons")
    ax.plot(gens, [d["predator"]["avg_hidden_complexity"] for d in data], color=palette["pred"], label="Predator avg hidden neurons")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg. hidden layer size")
    ax.set_title("Evolved Network Complexity Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_network_complexity.png"), dpi=130)
    plt.close(fig)

    # 3. Attention profile evolution (swarm)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for key, color in zip(["wall", "avatar_sight", "predator_sight", "pheromone"],
                           ["#7f8c8d", "#3498db", "#e74c3c", "#2ecc71"]):
        axes[0].plot(gens, [d["swarm"]["attention"][key] for d in data], label=key, color=color)
    axes[0].set_title("Swarm: What The Network Weighs Most")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Mean |input weight|")
    axes[0].legend(fontsize=8)

    for key, color in zip(["wall", "pheromone", "avatar_sight", "other_predator"],
                           ["#7f8c8d", "#2ecc71", "#3498db", "#f39c12"]):
        axes[1].plot(gens, [d["predator"]["attention"][key] for d in data], label=key, color=color)
    axes[1].set_title("Predator: What The Network Weighs Most")
    axes[1].set_xlabel("Generation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_attention_profiles.png"), dpi=130)
    plt.close(fig)

    # 4. Fusions & deaths
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(gens, [d["fusions"] for d in data], color="#9b59b6", alpha=0.6, label="Fusions")
    ax1.set_ylabel("Fusions", color="#9b59b6")
    ax1.set_xlabel("Generation")
    ax2 = ax1.twinx()
    ax2.plot(gens, [d["deaths"] for d in data], color="#e74c3c", linewidth=2, label="Deaths")
    ax2.set_ylabel("Deaths", color="#e74c3c")
    ax1.set_title("Hive Cohesion (Fusions) vs Predation Pressure (Deaths)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_fusions_vs_deaths.png"), dpi=130)
    plt.close(fig)

    # 5. Pheromone signal strength vs predator's attention to pheromone (the "go quiet" trade-off)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(gens, [d["swarm"]["avg_pheromone_signal"] for d in data], color="#2ecc71", label="Swarm avg pheromone output")
    ax1.set_ylabel("Swarm pheromone output", color="#2ecc71")
    ax1.set_xlabel("Generation")
    ax2 = ax1.twinx()
    ax2.plot(gens, [d["predator"]["attention"]["pheromone"] for d in data], color="#e67e22", label="Predator attention to pheromone")
    ax2.set_ylabel("Predator pheromone attention", color="#e67e22")
    ax1.set_title("Stigmergy Trade-off: Hive Signal Strength vs Predator Tracking")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_stigmergy_tradeoff.png"), dpi=130)
    plt.close(fig)

    # 6. Predator survival outcome per generation
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gens, [d["predators_alive_end"] for d in data], color="#e74c3c", drawstyle="steps-post")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Predators alive at match end")
    ax.set_title("Predator Survival Across Generations")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_predator_survival.png"), dpi=130)
    plt.close(fig)

    return [
        "01_arms_race_fitness.png", "02_network_complexity.png", "03_attention_profiles.png",
        "04_fusions_vs_deaths.png", "05_stigmergy_tradeoff.png", "06_predator_survival.png",
    ]


def summarize(data):
    gens = len(data)
    first, last = data[0], data[-1]
    swarm_gain = last["swarm"]["best_fitness"] - first["swarm"]["best_fitness"]
    pred_gain = last["predator"]["best_fitness"] - first["predator"]["best_fitness"]
    total_wipes = sum(1 for d in data if d["all_predators_killed"])
    avg_fusions = np.mean([d["fusions"] for d in data])
    avg_deaths = np.mean([d["deaths"] for d in data])
    complexity_growth_swarm = last["swarm"]["avg_hidden_complexity"] - first["swarm"]["avg_hidden_complexity"]
    complexity_growth_pred = last["predator"]["avg_hidden_complexity"] - first["predator"]["avg_hidden_complexity"]

    # correlation between swarm pheromone output and predator pheromone attention
    phero_signal = np.array([d["swarm"]["avg_pheromone_signal"] for d in data])
    pred_phero_attn = np.array([d["predator"]["attention"]["pheromone"] for d in data])
    corr = float(np.corrcoef(phero_signal, pred_phero_attn)[0, 1]) if gens > 2 else float("nan")

    return dict(
        gens=gens, swarm_gain=swarm_gain, pred_gain=pred_gain, total_wipes=total_wipes,
        avg_fusions=avg_fusions, avg_deaths=avg_deaths,
        complexity_growth_swarm=complexity_growth_swarm, complexity_growth_pred=complexity_growth_pred,
        stigmergy_corr=corr,
    )


def write_report(data, chart_files, out_dir, log_path):
    s = summarize(data)
    lines = []
    lines.append("# Co-Evolutionary Hive vs Predator Simulation — Results Report\n")
    lines.append(f"*Source log: `{os.path.basename(log_path)}` — {s['gens']} generations*\n")

    lines.append("## Abstract\n")
    lines.append(
        "This experiment evolves two competing neural populations inside a procedurally generated maze: "
        "a **swarm/hive** of avatars that fuse together for size-based survival and communicate via a learned "
        "pheromone field, and a **predator population** that hunts the swarm and co-evolves its own tactics in "
        "response. Both species use a population-based genetic algorithm with weight mutation, crossover, and "
        "NEAT-lite structural mutation (the hidden layer size itself can grow or shrink across generations).\n"
    )

    lines.append("## Methodology\n")
    lines.append("**Environment.** A procedurally generated maze (randomized depth-first search plus a loop-adding "
                  "pass to avoid single-path corridors) is regenerated every 15 generations to discourage "
                  "maze-specific overfitting.\n")
    lines.append("**Swarm learning.** 50 avatars are piloted by a population of 10 genomes (5 avatars per genome). "
                  "Avatars perceive walls, other avatars, the predator, and a local pheromone gradient in 4 "
                  "directions, plus their own fused size. Their 5th output controls how much pheromone they "
                  "deposit — coordination is a *learned* behavior, not a hardcoded rule.\n")
    lines.append("**Predator learning.** 5 predators are piloted by a population of 5 genomes (1 each), so every "
                  "predator genome is evaluated every generation. Predators perceive walls, the pheromone field, "
                  "direct sightlines to avatars, and nearby predators (to encourage spreading out rather than "
                  "clumping).\n")
    lines.append("**Evolution.** Both populations use tournament selection (k=3), uniform crossover, elitism "
                  "(top 2 genomes carried over), Gaussian weight mutation, and an 8% chance per offspring of a "
                  "structural mutation (hidden layer size ±1, clipped to [6, 32]).\n")
    lines.append("**Fitness.** Swarm genomes are credited for fusions caused by their avatars, survival time, and "
                  "a large bonus if one of their avatars grows big enough to kill a predator. Predator genomes are "
                  "credited for kills, proximity-based pursuit shaping, and survival time.\n")

    lines.append("## Results\n")

    lines.append("### 1. The Arms Race\n")
    lines.append(f"![arms race]({chart_files[0]})\n")
    lines.append(
        f"Over {s['gens']} generations, the swarm's best fitness moved by **{s['swarm_gain']:+.0f}** and the "
        f"predator's best fitness moved by **{s['pred_gain']:+.0f}**. "
        f"Predators fully wiped from the maze (swarm total victory) occurred in **{s['total_wipes']}** of "
        f"{s['gens']} generations. If the two curves rise together, that's the signature of genuine co-evolution "
        "rather than one side simply solving the other.\n"
    )

    lines.append("### 2. Network Complexity (NEAT-lite)\n")
    lines.append(f"![complexity]({chart_files[1]})\n")
    lines.append(
        f"Swarm network complexity changed by **{s['complexity_growth_swarm']:+.1f}** average hidden neurons; "
        f"predator complexity changed by **{s['complexity_growth_pred']:+.1f}**. Growth suggests evolution found "
        "value in representing more complex behavior; shrinkage suggests simpler reactive policies were "
        "sufficient (or that structural mutation drifted without strong pressure toward complexity).\n"
    )

    lines.append("### 3. What Each Species Learned to Pay Attention To\n")
    lines.append(f"![attention]({chart_files[2]})\n")
    lines.append(
        "Each line tracks the average magnitude of input weights per perception channel in the best genome each "
        "generation — a proxy for what the network has learned matters. A rising `predator_sight` line for the "
        "swarm means avatars are learning to actively react to the predator rather than move randomly; a rising "
        "`pheromone` line for the predator means it's learning to track the hive's own coordination signal.\n"
    )

    lines.append("### 4. Hive Cohesion vs Predation Pressure\n")
    lines.append(f"![fusions vs deaths]({chart_files[3]})\n")
    lines.append(
        f"Average fusions per generation: **{s['avg_fusions']:.1f}**. Average deaths per generation: "
        f"**{s['avg_deaths']:.1f}**. Fusion count is the swarm's main lever for creating a predator-killing giant; "
        "deaths are the predator's main lever for suppressing that strategy before it snowballs.\n"
    )

    lines.append("### 5. The Stigmergy Trade-off\n")
    lines.append(f"![stigmergy tradeoff]({chart_files[4]})\n")
    corr_txt = f"{s['stigmergy_corr']:.2f}" if s["stigmergy_corr"] == s["stigmergy_corr"] else "n/a"
    lines.append(
        f"Correlation between swarm pheromone output and predator attention to pheromone: **r = {corr_txt}**. "
        "The pheromone channel is a double-edged sword for the swarm: strong signals help avatars find each "
        "other to fuse, but the same signal is exactly what predators can evolve to track. A positive "
        "correlation here is evidence the predator population is specifically adapting to exploit the swarm's "
        "own communication channel — real predator-prey signal exploitation, the same dynamic seen in real "
        "biological stigmergy/eavesdropping arms races.\n"
    )

    lines.append("### 6. Predator Survival\n")
    lines.append(f"![predator survival]({chart_files[5]})\n")
    lines.append(
        "Tracks how many of the 5 predators were still alive at the end of each match — a direct measure of "
        "swarm dominance over time.\n"
    )

    lines.append("## Discussion\n")
    lines.append(
        "Unlike the v1 simulation (a single shared brain hill-climbing until the first predator kill, then "
        "exiting), this setup never 'finishes' — the swarm and predator are locked in a continuous arms race, "
        "which is the more realistic and more interesting regime for studying emergent multi-agent behavior. "
        "The pheromone/stigmergy mechanism in particular creates a genuinely open-ended tension: any coordination "
        "signal useful to the hive is also useful to something hunting the hive. Watching whether swarm pheromone "
        "output trends down over the run (evolving quieter, harder-to-track coordination) versus staying high "
        "(betting on fusing faster than the predator can capitalize on the signal) is one of the more interesting "
        "things to watch in longer runs.\n"
    )

    lines.append("## Limitations & Simplifications\n")
    lines.append(
        "- **NEAT-lite, not full NEAT**: only hidden layer *size* evolves, not arbitrary connection topology "
        "(no innovation numbers, no per-connection add/remove). Full NEAT would allow non-layered, sparse, or "
        "skip-connection architectures to emerge.\n"
        "- **Credit assignment is heuristic**: when avatars fuse, the surviving avatar's genome gets all the "
        "credit; this is a reasonable approximation but not a rigorous multi-agent credit assignment scheme.\n"
        "- **Single pheromone channel**: real ant-colony stigmergy often uses multiple chemical signals "
        "(food trail vs. danger trail); this simulation uses one learned scalar per cell.\n"
    )

    lines.append("## Suggested Next Experiments\n")
    lines.append(
        "1. Run for 500+ generations and watch whether the arms-race gap oscillates (predator-prey cycles) "
        "or one side runs away.\n"
        "2. Add a second pheromone channel (e.g., a distinct 'danger' signal deposited on death) and see if "
        "the hive learns to separate 'gather here' from 'flee this area' signals.\n"
        "3. Log per-genome lineage (parent IDs) to build a phylogenetic tree of which strategies dominated.\n"
    )

    report_path = os.path.join(out_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--out", default="report")
    args = parser.parse_args()

    data = load_log(args.log_path)
    chart_files = make_charts(data, args.out)
    report_path = write_report(data, chart_files, args.out, args.log_path)
    print(f"Report written to {report_path}")
    print(f"Charts written to {args.out}/")


if __name__ == "__main__":
    main()
