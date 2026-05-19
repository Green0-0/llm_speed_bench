#!/usr/bin/env python3
import os
import glob
import json
import time
from collections import defaultdict

# Try importing matplotlib for plotting, warn if not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def make_markdown_table(headers, rows):
    header_str = "| " + " | ".join(headers) + " |\n"
    separator_str = "| " + " | ".join(["---"] * len(headers)) + " |\n"
    rows_str = ""
    for row in rows:
        rows_str += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    return header_str + separator_str + rows_str

def get_base_config_and_spec(config_id):
    if config_id.endswith("-df"):
        return config_id[:-3], "dflash"
    elif config_id.endswith("-mtp"):
        return config_id[:-4], "mtp"
    else:
        return config_id, "base"

def generate_plots(gpu_name, gpu_safe_name, aggregated_configs, sorted_configs, sorted_topics, sorted_tasks, task_to_topic, engine_speedups, spec_speedups, quant_speedups):
    if not MATPLOTLIB_AVAILABLE:
        print(f"Warning: matplotlib is not installed. Skipping graph generation for GPU: {gpu_name}")
        return

    # Setup premium styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    COLORS = ['#3A86C8', '#E65C00', '#2CA02C', '#9467BD', '#8C564B', '#E377C2', '#BCBD22', '#17BECF']
    
    # 1. OVERALL SPEED COMPARISON PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = list(range(len(sorted_configs)))
    width = 0.35
    
    agg_speeds = [aggregated_configs[cfg]["overall_aggregate_throughput"] for cfg in sorted_configs]
    mean_speeds = [aggregated_configs[cfg]["overall_mean_throughput"] for cfg in sorted_configs]
    
    rects1 = ax.bar([val - width/2 for val in x], agg_speeds, width, label='Aggregate Throughput', color='#3A86C8', alpha=0.9, edgecolor='none')
    rects2 = ax.bar([val + width/2 for val in x], mean_speeds, width, label='Mean Decode Throughput', color='#E65C00', alpha=0.9, edgecolor='none')
    
    ax.set_ylabel('Throughput (tokens/second)', fontsize=12, fontweight='bold', color='#333333')
    ax.set_title(f'Overall Throughput Comparison\nGPU: {gpu_name}', fontsize=14, fontweight='bold', pad=20, color='#111111')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_configs, rotation=30, ha='right', fontsize=10)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd', loc='upper left')
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#444444')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plot_path = f"analysis/overall_comparison_{gpu_safe_name}.png"
    plt.savefig(plot_path, dpi=300, facecolor='#fdfdfd')
    plt.close()

    # 2. TOPIC SPEED COMPARISON PLOT
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = list(range(len(sorted_topics)))
    width = 0.8 / max(1, len(sorted_configs))
    
    for idx, config_key in enumerate(sorted_configs):
        topic_speeds = []
        for topic in sorted_topics:
            topic_speeds.append(aggregated_configs[config_key]["topics"].get(topic, {}).get("aggregate", 0.0))
        
        offset = (idx - len(sorted_configs)/2 + 0.5) * width
        color = COLORS[idx % len(COLORS)]
        ax.bar([val + offset for val in x], topic_speeds, width, label=config_key, color=color, alpha=0.85, edgecolor='none')
        
    ax.set_ylabel('Aggregate Throughput (tokens/second)', fontsize=12, fontweight='bold', color='#333333')
    ax.set_title(f'Throughput by Topic (Merged Category Performance)\nGPU: {gpu_name}', fontsize=14, fontweight='bold', pad=20, color='#111111')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_topics, rotation=15, ha='right', fontsize=11)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd', title='Configurations')
    
    plt.tight_layout()
    plot_path = f"analysis/topic_comparison_{gpu_safe_name}.png"
    plt.savefig(plot_path, dpi=300, facecolor='#fdfdfd')
    plt.close()

    # 3. TASK (PROMPT ID) SPEED COMPARISON PLOT (Horizontal)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y = list(range(len(sorted_tasks)))
    height = 0.8 / max(1, len(sorted_configs))
    
    for idx, config_key in enumerate(sorted_configs):
        task_speeds = []
        for task_id in sorted_tasks:
            task_speeds.append(aggregated_configs[config_key]["tasks"].get(task_id, {}).get("aggregate", 0.0))
            
        offset = (idx - len(sorted_configs)/2 + 0.5) * height
        color = COLORS[idx % len(COLORS)]
        ax.barh([val + offset for val in y], task_speeds, height, label=config_key, color=color, alpha=0.85, edgecolor='none')
        
    ax.set_xlabel('Aggregate Throughput (tokens/second)', fontsize=12, fontweight='bold', color='#333333')
    ax.set_ylabel('Task (Prompt ID)', fontsize=12, fontweight='bold', color='#333333')
    ax.set_title(f'Throughput by Task (Individual Prompt Performance)\nGPU: {gpu_name}', fontsize=14, fontweight='bold', pad=20, color='#111111')
    ax.set_yticks(y)
    
    task_labels = [f"{tid} ({task_to_topic.get(tid, 'Unknown')[:12]}...)" for tid in sorted_tasks]
    ax.set_yticklabels(task_labels, fontsize=10)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd', title='Configurations')
    
    plt.tight_layout()
    plot_path = f"analysis/task_comparison_{gpu_safe_name}.png"
    plt.savefig(plot_path, dpi=300, facecolor='#fdfdfd')
    plt.close()

    # 4. ENGINE SPEEDUP PLOT (SGLang vs vLLM)
    if engine_speedups:
        fig, ax = plt.subplots(figsize=(10, 5))
        configs_to_plot = sorted(list(engine_speedups.keys()))
        ratios = [engine_speedups[c] for c in configs_to_plot]
        
        # Color green for speedup, red for slowdown
        bar_colors = ['#2CA02C' if r >= 1.0 else '#D62728' for r in ratios]
        
        x = list(range(len(configs_to_plot)))
        rects = ax.bar(x, ratios, width=0.5, color=bar_colors, alpha=0.85, edgecolor='none')
        ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=1.5, label='Performance Parity')
        
        ax.set_ylabel('Speedup Ratio (SGLang / vLLM)', fontsize=12, fontweight='bold', color='#333333')
        ax.set_title(f'Inference Engine Speedup (SGLang vs vLLM)\nGPU: {gpu_name}', fontsize=14, fontweight='bold', pad=20, color='#111111')
        ax.set_xticks(x)
        ax.set_xticklabels(configs_to_plot, rotation=25, ha='right', fontsize=10)
        ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2fx}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3) if height >= 1.0 else (0, -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 1.0 else 'top',
                        fontsize=9, fontweight='bold', color='#333333')
                        
        plt.tight_layout()
        plot_path = f"analysis/engine_speedup_{gpu_safe_name}.png"
        plt.savefig(plot_path, dpi=300, facecolor='#fdfdfd')
        plt.close()

    # 5. SPECULATIVE SPEEDUP PLOT
    has_spec_data = any(len(spec_speedups[eng]) > 0 for eng in spec_speedups)
    if has_spec_data:
        flat_spec_data = []
        for engine in sorted(spec_speedups.keys()):
            for base_id in sorted(spec_speedups[engine].keys()):
                flat_spec_data.append((engine, base_id, spec_speedups[engine][base_id]))
                
        if flat_spec_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = list(range(len(flat_spec_data)))
            labels = [f"{eng}\n{base}" for eng, base, _ in flat_spec_data]
            
            dflash_ratios = [data[2].get('dflash', 0.0) for data in flat_spec_data]
            mtp_ratios = [data[2].get('mtp', 0.0) for data in flat_spec_data]
            
            width = 0.35
            ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=1.5, label='Baseline (No Spec)')
            
            rects_df = ax.bar([val - width/2 for val in x], dflash_ratios, width, label='DFlash Speculative', color='#2CA02C', alpha=0.9, edgecolor='none')
            rects_mtp = ax.bar([val + width/2 for val in x], mtp_ratios, width, label='MTP Speculative', color='#9467BD', alpha=0.9, edgecolor='none')
            
            ax.set_ylabel('Speedup Factor (Spec / Base)', fontsize=12, fontweight='bold', color='#333333')
            ax.set_title(f'Speculative Decoding Throughput Speedup\nGPU: {gpu_name}', fontsize=14, fontweight='bold', pad=20, color='#111111')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
            
            for rects in [rects_df, rects_mtp]:
                for rect in rects:
                    height = rect.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.2fx}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
            
            plt.tight_layout()
            plot_path = f"analysis/speculative_speedup_{gpu_safe_name}.png"
            plt.savefig(plot_path, dpi=300, facecolor='#fdfdfd')
            plt.close()

    # 6. QUANTIZATION SPEEDUP PLOT
    has_quant_data = any(len(quant_speedups[eng]) > 0 for eng in quant_speedups)
    if has_quant_data:
        flat_quant_data = []
        for engine in sorted(quant_speedups.keys()):
            for base_id in sorted(quant_speedups[engine].keys()):
                flat_quant_data.append((engine, base_id, quant_speedups[engine][base_id]))
                
        if flat_quant_data:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = list(range(len(flat_quant_data)))
            labels = [f"{eng}\n{base}" for eng, base, _ in flat_quant_data]
            ratios = [data[2] for data in flat_quant_data]
            
            rects = ax.bar(x, ratios, width=0.4, color='#17BECF', alpha=0.9, edgecolor='none')
            ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=1.5, label='Baseline (BF16)')
            
            ax.set_ylabel('Speedup Factor (AWQ / BF16)', fontsize=12, fontweight='bold', color='#333333')
            ax.set_title(f'AWQ Quantization Throughput Speedup\nGPU: {gpu_name}', fontsize=14, fontweight='bold', pad=20, color='#111111')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
            
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2fx}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
                            
            plt.tight_layout()
            plot_path = f"analysis/quantization_speedup_{gpu_safe_name}.png"
            plt.savefig(plot_path, dpi=300, facecolor='#fdfdfd')
            plt.close()

    print(f"Plots saved for GPU '{gpu_name}' in the 'analysis' folder.")

def main():
    # Make directories if not exist
    os.makedirs("analysis", exist_ok=True)

    # Find all JSON files in results
    result_files = glob.glob("results/*.json")
    if not result_files:
        print("No result files found in the 'results' folder.")
        return

    print(f"Found {len(result_files)} result files. Processing...")

    # Group data by GPU and then by Config (engine + config_id)
    # Structure: gpu_name -> config_key -> list of run_data
    gpu_groups = defaultdict(lambda: defaultdict(list))

    for filepath in result_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                run_data = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        gpu_name = run_data.get("gpu_name", "Unknown GPU")
        engine = run_data.get("engine", "unknown")
        config_id = run_data.get("config_id", "unknown")
        config_key = f"{engine}_{config_id}"

        gpu_groups[gpu_name][config_key].append(run_data)

    # Dictionary to collect all structured aggregated data across GPUs to write to JSON
    all_gpu_aggregated_data = {}

    # Accumulate report contents
    report_md = "# LLM Speed Benchmark Performance Analysis Report\n"
    report_md += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_md += "This report aggregates performance metrics from benchmark runs, separating findings per GPU, per configuration, per topic category, and per individual task (prompt ID).\n\n"

    for gpu_name, configs in gpu_groups.items():
        print(f"\nProcessing GPU: {gpu_name}")
        gpu_safe_name = gpu_name.replace("/", "_").replace(" ", "_").replace("-", "_").lower()

        # Aggregated stats for this GPU
        # config_key -> stats dict
        aggregated_configs = {}

        # All topics and task IDs encountered for this GPU (to align tables/plots)
        all_topics = set()
        all_tasks = set()
        task_to_topic = {}

        for config_key, runs in configs.items():
            # We want to aggregate metrics across all runs of this config
            run_aggregates = []
            run_means = []
            run_latencies = []
            
            # topic -> list of averages (one per run)
            topic_aggs_runs = defaultdict(list)
            topic_means_runs = defaultdict(list)
            
            # task_id -> list of values (one per run)
            task_aggs_runs = defaultdict(list)
            task_means_runs = defaultdict(list)

            for run in runs:
                results = run.get("results", [])
                if not results:
                    continue

                # Run-level values
                latencies = [r["avg_latency"] for r in results if "avg_latency" in r]
                mean_tps = [r["avg_mean_throughput"] for r in results if "avg_mean_throughput" in r]
                agg_tps = [r["avg_aggregate_throughput"] for r in results if "avg_aggregate_throughput" in r]

                if latencies:
                    run_latencies.append(mean(latencies))
                if mean_tps:
                    run_means.append(mean(mean_tps))
                if agg_tps:
                    run_aggregates.append(mean(agg_tps))

                # Group by topic in this run
                topic_groups = defaultdict(list)
                for r in results:
                    topic = r.get("topic", "Unknown")
                    all_topics.add(topic)
                    topic_groups[topic].append(r)

                for topic, items in topic_groups.items():
                    t_means = [item["avg_mean_throughput"] for item in items if "avg_mean_throughput" in item]
                    t_aggs = [item["avg_aggregate_throughput"] for item in items if "avg_aggregate_throughput" in item]
                    if t_means:
                        topic_means_runs[topic].append(mean(t_means))
                    if t_aggs:
                        topic_aggs_runs[topic].append(mean(t_aggs))

                # Group by task in this run
                for r in results:
                    task_id = r.get("prompt_id", "Unknown")
                    topic = r.get("topic", "Unknown")
                    all_tasks.add(task_id)
                    task_to_topic[task_id] = topic

                    if "avg_mean_throughput" in r:
                        task_means_runs[task_id].append(r["avg_mean_throughput"])
                    if "avg_aggregate_throughput" in r:
                        task_aggs_runs[task_id].append(r["avg_aggregate_throughput"])

            if not run_aggregates:
                continue

            # Average across runs
            aggregated_configs[config_key] = {
                "overall_aggregate_throughput": mean(run_aggregates),
                "overall_mean_throughput": mean(run_means),
                "overall_latency": mean(run_latencies),
                "topics": {
                    topic: {
                        "aggregate": mean(topic_aggs_runs[topic]) if topic in topic_aggs_runs else 0.0,
                        "mean": mean(topic_means_runs[topic]) if topic in topic_means_runs else 0.0,
                    }
                    for topic in sorted(topic_aggs_runs.keys())
                },
                "tasks": {
                    task_id: {
                        "aggregate": mean(task_aggs_runs[task_id]) if task_id in task_aggs_runs else 0.0,
                        "mean": mean(task_means_runs[task_id]) if task_id in task_means_runs else 0.0,
                    }
                    for task_id in sorted(task_aggs_runs.keys())
                }
            }

        if not aggregated_configs:
            print(f"No valid data to aggregate for GPU: {gpu_name}")
            continue

        sorted_configs = sorted(aggregated_configs.keys())
        sorted_topics = sorted(list(all_topics))
        sorted_tasks = sorted(list(all_tasks))

        # --- Compute Speedup Comparisons ---
        # 1. Engine comparison speedups
        engine_speedups = {}
        for config_key, stats in aggregated_configs.items():
            if not config_key.startswith("sglang_"):
                continue
            config_id = config_key[7:]
            vllm_key = f"vllm_{config_id}"
            if vllm_key in aggregated_configs:
                sglang_tp = stats["overall_aggregate_throughput"]
                vllm_tp = aggregated_configs[vllm_key]["overall_aggregate_throughput"]
                if vllm_tp > 0:
                    engine_speedups[config_id] = sglang_tp / vllm_tp

        # 2. Speculative decoding speedups
        spec_speedups = defaultdict(dict)
        for config_key, stats in aggregated_configs.items():
            if "_" not in config_key:
                continue
            engine, config_id = config_key.split("_", 1)
            base_id, spec_method = get_base_config_and_spec(config_id)
            
            if spec_method != "base":
                base_key = f"{engine}_{base_id}"
                if base_key in aggregated_configs:
                    base_tp = aggregated_configs[base_key]["overall_aggregate_throughput"]
                    if base_tp > 0:
                        spec_tp = stats["overall_aggregate_throughput"]
                        spec_speedups[engine].setdefault(base_id, {})[spec_method] = spec_tp / base_tp

        # 3. Quantization speedups
        quant_speedups = defaultdict(dict)
        for config_key, stats in aggregated_configs.items():
            if "_" not in config_key:
                continue
            engine, config_id = config_key.split("_", 1)
            base_id, spec_method = get_base_config_and_spec(config_id)
            if spec_method != "base":
                continue
            
            if config_id.endswith("-awq"):
                bf16_id = config_id[:-4]
                bf16_key = f"{engine}_{bf16_id}"
                if bf16_key in aggregated_configs:
                    awq_tp = stats["overall_aggregate_throughput"]
                    bf16_tp = aggregated_configs[bf16_key]["overall_aggregate_throughput"]
                    if bf16_tp > 0:
                        quant_speedups[engine][bf16_id] = awq_tp / bf16_tp

        # Save aggregated JSON data
        all_gpu_aggregated_data[gpu_name] = {
            "configs": aggregated_configs,
            "sorted_configs": sorted_configs,
            "sorted_topics": sorted_topics,
            "sorted_tasks": sorted_tasks,
            "speedups": {
                "engine": engine_speedups,
                "speculative": {eng: dict(d) for eng, d in spec_speedups.items()},
                "quantization": {eng: dict(d) for eng, d in quant_speedups.items()}
            }
        }

        # Generate plots
        generate_plots(gpu_name, gpu_safe_name, aggregated_configs, sorted_configs, sorted_topics, sorted_tasks, task_to_topic, engine_speedups, spec_speedups, quant_speedups)

        # Build Markdown Section for this GPU
        report_md += f"## GPU: {gpu_name}\n\n"
        
        # 1. Overall comparison table
        report_md += "### 1. Overall Performance Summary\n"
        report_md += "Averages across all prompts and runs for each server configuration.\n\n"
        
        overall_headers = ["Configuration", "Overall Aggregate Throughput (t/s)", "Overall Mean Decode Throughput (t/s)", "Overall Latency (s)"]
        overall_rows = []
        for cfg in sorted_configs:
            stats = aggregated_configs[cfg]
            overall_rows.append([
                cfg,
                f"{stats['overall_aggregate_throughput']:.2f}",
                f"{stats['overall_mean_throughput']:.2f}",
                f"{stats['overall_latency']:.3f}"
            ])
        report_md += make_markdown_table(overall_headers, overall_rows) + "\n"

        # 2. Engine comparison table
        if engine_speedups:
            report_md += "### 2. Inference Engine Comparison (SGLang vs vLLM)\n"
            report_md += "Relative speedup of SGLang compared to vLLM on identical model configurations. Ratios > 1.0 indicate SGLang is faster.\n\n"
            engine_headers = ["Configuration ID", "SGLang Throughput (t/s)", "vLLM Throughput (t/s)", "Speedup Ratio"]
            engine_rows = []
            for config_id in sorted(engine_speedups.keys()):
                sglang_tp = aggregated_configs[f"sglang_{config_id}"]["overall_aggregate_throughput"]
                vllm_tp = aggregated_configs[f"vllm_{config_id}"]["overall_aggregate_throughput"]
                speedup = engine_speedups[config_id]
                engine_rows.append([
                    config_id,
                    f"{sglang_tp:.2f}",
                    f"{vllm_tp:.2f}",
                    f"**{speedup:.2fx}**"
                ])
            report_md += make_markdown_table(engine_headers, engine_rows) + "\n"

        # 3. Speculative decoding comparison
        has_spec_data = any(len(spec_speedups[eng]) > 0 for eng in spec_speedups)
        if has_spec_data:
            report_md += "### 3. Speculative Decoding Performance Impact\n"
            report_md += "Throughput speedup provided by speculative decoding methods (DFlash and MTP) relative to their non-speculative base models.\n\n"
            spec_headers = ["Engine", "Base Config", "Method", "Base Throughput (t/s)", "Spec Throughput (t/s)", "Speedup Ratio"]
            spec_rows = []
            for engine in sorted(spec_speedups.keys()):
                for base_id in sorted(spec_speedups[engine].keys()):
                    base_key = f"{engine}_{base_id}"
                    base_tp = aggregated_configs[base_key]["overall_aggregate_throughput"]
                    for method, speedup in sorted(spec_speedups[engine][base_id].items()):
                        spec_id = f"{base_id}-df" if method == "dflash" else f"{base_id}-mtp"
                        spec_key = f"{engine}_{spec_id}"
                        if spec_key in aggregated_configs:
                            spec_tp = aggregated_configs[spec_key]["overall_aggregate_throughput"]
                            spec_rows.append([
                                engine,
                                base_id,
                                method.upper(),
                                f"{base_tp:.2f}",
                                f"{spec_tp:.2f}",
                                f"**{speedup:.2fx}**"
                            ])
            if spec_rows:
                report_md += make_markdown_table(spec_headers, spec_rows) + "\n"

        # 4. Quantization comparison
        has_quant_data = any(len(quant_speedups[eng]) > 0 for eng in quant_speedups)
        if has_quant_data:
            report_md += "### 4. AWQ Quantization Speedup Impact\n"
            report_md += "Performance benefit of 4-bit AWQ quantization compared to baseline BF16 precision.\n\n"
            quant_headers = ["Engine", "Model", "BF16 Throughput (t/s)", "AWQ Throughput (t/s)", "Speedup Ratio"]
            quant_rows = []
            for engine in sorted(quant_speedups.keys()):
                for base_id in sorted(quant_speedups[engine].keys()):
                    bf16_key = f"{engine}_{base_id}"
                    awq_key = f"{engine}_{base_id}-awq"
                    bf16_tp = aggregated_configs[bf16_key]["overall_aggregate_throughput"]
                    awq_tp = aggregated_configs[awq_key]["overall_aggregate_throughput"]
                    speedup = quant_speedups[engine][base_id]
                    quant_rows.append([
                        engine,
                        base_id,
                        f"{bf16_tp:.2f}",
                        f"{awq_tp:.2f}",
                        f"**{speedup:.2fx}**"
                    ])
            if quant_rows:
                report_md += make_markdown_table(quant_headers, quant_rows) + "\n"

        # 5. Topic summary table
        report_md += "### 5. Performance by Topic Tag (Averaged)\n"
        report_md += "Speeds for prompts grouped and averaged by their topic categories.\n\n"
        
        topic_headers = ["Topic / Category"] + [f"{cfg} (Agg)" for cfg in sorted_configs]
        topic_rows = []
        for topic in sorted_topics:
            row = [topic]
            for cfg in sorted_configs:
                speed = aggregated_configs[cfg]["topics"].get(topic, {}).get("aggregate", 0.0)
                row.append(f"{speed:.2f} t/s")
            topic_rows.append(row)
        report_md += make_markdown_table(topic_headers, topic_rows) + "\n"

        # 6. Task summary table
        report_md += "### 6. Performance by Task (Individual Prompt ID)\n"
        report_md += "Speeds for each individual prompt ID under each server configuration.\n\n"
        
        task_headers = ["Task ID", "Topic Category"] + [f"{cfg} (Agg)" for cfg in sorted_configs]
        task_rows = []
        for task_id in sorted_tasks:
            row = [task_id, task_to_topic.get(task_id, "Unknown")]
            for cfg in sorted_configs:
                speed = aggregated_configs[cfg]["tasks"].get(task_id, {}).get("aggregate", 0.0)
                row.append(f"{speed:.2f} t/s")
            task_rows.append(row)
        report_md += make_markdown_table(task_headers, task_rows) + "\n"

        # 7. Graphs
        report_md += "### 7. Comparison Graphs\n\n"
        if MATPLOTLIB_AVAILABLE:
            report_md += f"#### Overall Throughput Comparison\n"
            report_md += f"![Overall Comparison](overall_comparison_{gpu_safe_name}.png)\n\n"
            
            if engine_speedups:
                report_md += f"#### SGLang vs vLLM Speedup Comparison\n"
                report_md += f"![Engine Speedup](engine_speedup_{gpu_safe_name}.png)\n\n"
                
            if has_spec_data:
                report_md += f"#### Speculative Decoding Speedup Comparison\n"
                report_md += f"![Speculative Speedup](speculative_speedup_{gpu_safe_name}.png)\n\n"
                
            if has_quant_data:
                report_md += f"#### AWQ Quantization Speedup Comparison\n"
                report_md += f"![Quantization Speedup](quantization_speedup_{gpu_safe_name}.png)\n\n"
                
            report_md += f"#### Throughput by Topic\n"
            report_md += f"![Topic Comparison](topic_comparison_{gpu_safe_name}.png)\n\n"
            report_md += f"#### Throughput by Task (Individual Prompts)\n"
            report_md += f"![Task Comparison](task_comparison_{gpu_safe_name}.png)\n\n"
        else:
            report_md += "*Note: Graphs could not be generated because matplotlib is not installed in the execution environment.*\n\n"
        
        report_md += "---\n\n"

    # Write summary report in JSON
    json_report_path = "analysis/summary_report.json"
    try:
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(all_gpu_aggregated_data, f, indent=2)
        print(f"Summary JSON saved to {json_report_path}")
    except Exception as e:
        print(f"Error saving summary JSON: {e}")

    # Write combined Markdown report
    md_report_path = "analysis/performance_report.md"
    try:
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Performance report saved to {md_report_path}")
    except Exception as e:
        print(f"Error saving markdown report: {e}")

if __name__ == "__main__":
    main()
