import argparse
import json
import os
import subprocess
import sys
import time
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
from tqdm import tqdm

MODEL_CONFIGS = {
    # Qwen 3.5
    "qwen35-9b": {
        "model_name": "Qwen/Qwen3.5-9B",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "qwen35-9b-awq": {
        "model_name": "QuantTrio/Qwen3.5-9B-AWQ",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "qwen35-27b-awq": {
        "model_name": "QuantTrio/Qwen3.5-27B-AWQ",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "qwen35-35b-awq": {
        "model_name": "QuantTrio/Qwen3.5-35B-A3B-AWQ",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "qwen35-9b-awq-df": {
        "model_name": "QuantTrio/Qwen3.5-9B-AWQ",
        "spec_method": "dflash",
        "spec_draft_model": "z-lab/Qwen3.5-9B-DFlash",
        "spec_tokens": 2,
    },
    "qwen35-27b-awq-df": {
        "model_name": "QuantTrio/Qwen3.5-27B-AWQ",
        "spec_method": "dflash",
        "spec_draft_model": "z-lab/Qwen3.5-27B-DFlash",
        "spec_tokens": 2,
    },
    "qwen35-35b-awq-df": {
        "model_name": "QuantTrio/Qwen3.5-35B-A3B-AWQ",
        "spec_method": "dflash",
        "spec_draft_model": "z-lab/Qwen3.5-35B-A3B-DFlash",
        "spec_tokens": 2,
    },
    "qwen35-9b-awq-mtp": {
        "model_name": "QuantTrio/Qwen3.5-9B-AWQ",
        "spec_method": "mtp",
        "spec_draft_model": None,
        "spec_tokens": 1,
    },
    "qwen35-27b-awq-mtp": {
        "model_name": "QuantTrio/Qwen3.5-27B-AWQ",
        "spec_method": "mtp",
        "spec_draft_model": None,
        "spec_tokens": 1,
    },
    "qwen35-35b-awq-mtp": {
        "model_name": "QuantTrio/Qwen3.5-35B-A3B-AWQ",
        "spec_method": "mtp",
        "spec_draft_model": None,
        "spec_tokens": 1,
    },
    # Gemma 4
    "gemma4-e4b": {
        "model_name": "google/gemma-4-E4B-it",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "gemma4-e4b-awq": {
        "model_name": "cyankiwi/gemma-4-E4B-it-AWQ-INT4",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "gemma4-26b-awq": {
        "model_name": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "gemma4-31b-awq": {
        "model_name": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "gemma4-26b-awq-df": {
        "model_name": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "spec_method": "dflash",
        "spec_draft_model": "z-lab/gemma-4-26B-A4B-it-DFlash",
        "spec_tokens": 2,
    },
    "gemma4-31b-awq-df": {
        "model_name": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
        "spec_method": "dflash",
        "spec_draft_model": "z-lab/gemma-4-31B-it-DFlash",
        "spec_tokens": 2,
    },
    "gemma4-e4b-awq-mtp": {
        "model_name": "cyankiwi/gemma-4-E4B-it-AWQ-INT4",
        "spec_method": "mtp",
        "spec_draft_model": "google/gemma-4-E4B-it-assistant",
        "spec_tokens": 1,
    },
    "gemma4-26b-awq-mtp": {
        "model_name": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "spec_method": "mtp",
        "spec_draft_model": "google/gemma-4-26B-A4B-it-assistant",
        "spec_tokens": 1,
    },
    "gemma4-31b-awq-mtp": {
        "model_name": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
        "spec_method": "mtp",
        "spec_draft_model": "google/gemma-4-31B-it-assistant",
        "spec_tokens": 1,
    },
    # Other Models
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-8B",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "qwen3-8b-awq-df": {
        "model_name": "Qwen/Qwen3-8B-AWQ",
        "spec_method": "dflash",
        "spec_draft_model": "z-lab/Qwen3-8B-DFlash-b16",
        "spec_tokens": 2,
    },
    "qwen3-8b-awq": {
        "model_name": "Qwen/Qwen3-8B-AWQ",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "qwen25-7b-awq": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "llama3-8b": {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "llama3-8b-awq-df": {
        "model_name": "casperhansen/llama-3-8b-instruct-awq",
        "spec_method": "dflash",
        "spec_draft_model": "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
        "spec_tokens": 2,
    },
    "llama3-8b-awq": {
        "model_name": "casperhansen/llama-3-8b-instruct-awq",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "granite-8b": {
        "model_name": "ibm-granite/granite-4.1-8b",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "granite-8b-awq": {
        "model_name": "cyankiwi/granite-4.1-8b-AWQ-INT4",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "granite-30b-awq": {
        "model_name": "cyankiwi/granite-4.1-30b-AWQ-INT4",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "gpt-oss-20b": {
        "model_name": "openai/gpt-oss-20b",
        "spec_method": None,
        "spec_draft_model": None,
        "spec_tokens": None,
    },
    "gpt-oss-20b-df": {
        "model_name": "openai/gpt-oss-20b",
        "spec_method": "dflash",
        "spec_draft_model": "z-lab/gpt-oss-20b-DFlash",
        "spec_tokens": 2,
    },
}

def launch_vllm_server(model_name, spec_method, spec_draft_model, spec_tokens, tp_size, port):
    if model_name == "Qwen/Qwen3.5-9B":
        mem_util = 0.75
    else:
        mem_util = 0.9
    cmd = [
        "vllm", "serve", model_name,
        "--port", str(port),
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", str(mem_util),
        "--max-model-len", "2048",
        "--max-num-seqs", "256",
        "--max-num-batched-tokens", "4096",
        "--disable-uvicorn-access-log",
    ]

    if spec_method == "dflash":
        spec_cfg = {
            "method": "dflash",
            "model": spec_draft_model,
            "num_speculative_tokens": spec_tokens,
        }
        cmd += ["--speculative-config", json.dumps(spec_cfg)]
    elif spec_method == "mtp":
        spec_cfg = {
            "method": "mtp",
            "num_speculative_tokens": spec_tokens,  # this will be 1
        }
        if spec_draft_model:
            spec_cfg["model"] = spec_draft_model
        cmd += ["--speculative-config", json.dumps(spec_cfg)]

    print("\nLaunching vLLM server...")
    print(" ".join(cmd))

    proc = subprocess.Popen(cmd)

    print("Waiting for vLLM server to start (this may take a few minutes)...")
    health_url = f"http://localhost:{port}/v1/models"

    for _ in range(600):
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print("vLLM server is ready.\n")
                return proc
        except Exception:
            pass
        time.sleep(2)

    proc.terminate()
    raise RuntimeError("vLLM server failed to start within the timeout.")


def generate_vllm(port, model_name, prompt, temperature, max_tokens, verbose=False):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    response.raise_for_status()
    data = response.json()

    choice = data["choices"][0]
    msg = choice.get("message", {})
    
    if verbose:
        print("finish_reason:", choice.get("finish_reason"))
        print("matched_stop:", choice.get("matched_stop"))
        print("content:", repr(msg.get("content")))
        print("reasoning_content:", repr(msg.get("reasoning_content")))

    text = msg.get("content") or ""
    tokens = data["usage"]["completion_tokens"]
    return text, tokens


def run_detailed_benchmark(args):
    config = MODEL_CONFIGS[args.config_id]
    model_name = config["model_name"]
    spec_method = config["spec_method"]
    spec_draft_model = config["spec_draft_model"]
    spec_tokens = config["spec_tokens"]

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")

    print("=" * 80)
    print("CONFIGURATION DETAILS:")
    print(f"  Config ID:        {args.config_id}")
    print(f"  Model Name:       {model_name}")
    print(f"  Spec Method:      {spec_method or 'None'}")
    print(f"  Spec Draft Model: {spec_draft_model or 'None'}")
    print(f"  Spec Tokens:      {spec_tokens or 'None'}")
    print(f"  Concurrent BSZ:   {args.bsz}")
    print(f"  Runs per Prompt:  {args.runs_per_prompt}")
    print(f"  Max Tokens:       {args.max_tokens}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  GPU Count (TP):   {num_gpus}")
    print("=" * 80)

    dataset_file = "data/lite.json" if args.lite else "data/full.json"
    print(f"Loading dataset from: {dataset_file}")
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        prompts = [item["prompt"] for item in dataset]
    except FileNotFoundError:
        print(f"Error: {dataset_file} not found. Ensure the dataset exists.")
        sys.exit(1)

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    server_proc = launch_vllm_server(
        model_name=model_name,
        spec_method=spec_method,
        spec_draft_model=spec_draft_model,
        spec_tokens=spec_tokens,
        tp_size=num_gpus,
        port=args.port,
    )

    list_of_prompt_results = []
    try:
        print("Warmup...")
        for p in prompts[:1]:
            generate_vllm(args.port, model_name, p, temperature=0.0, max_tokens=32)

        print("\n" + "=" * 80)
        print(
            f"Benchmark Running... (BSZ={args.bsz}, TEMP={args.temperature}, "
            f"TOKENS={args.max_tokens}, RUNS={args.runs_per_prompt}, "
            f"CONFIG_ID={args.config_id}, MODEL={model_name}, TP={num_gpus})"
        )
        print("=" * 80)

        for i, item in tqdm(enumerate(dataset, 1), total=len(dataset)):
            prompt = item["prompt"]
            topic = item.get("topic", "Unknown")
            prompt_id = item.get("id", f"prompt_{i}")

            latencies = []
            mean_throughputs = []
            aggregate_throughputs = []
            token_counts = []

            for _ in tqdm(range(args.runs_per_prompt), leave=False):
                start = time.time()
                total_tokens = 0
                req_speeds = []

                def fetch_req():
                    t0 = time.time()
                    _, num_tokens = generate_vllm(
                        args.port, 
                        model_name, 
                        prompt, 
                        args.temperature, 
                        args.max_tokens, 
                        False
                    )
                    t1 = time.time()
                    return num_tokens, t1 - t0
                
                with ThreadPoolExecutor(max_workers=args.bsz) as executor:
                    futures = [executor.submit(fetch_req) for _ in range(args.bsz)]
                    
                    for future in as_completed(futures):
                        num_tokens, req_time = future.result()
                        total_tokens += num_tokens
                        if req_time > 0:
                            req_speeds.append(num_tokens / req_time)

                batch_time = time.time() - start
                
                mean_req_throughput = mean(req_speeds) if req_speeds else 0.0
                
                aggregate_batch_throughput = total_tokens / batch_time if batch_time > 0 else 0.0

                latencies.append(batch_time)
                mean_throughputs.append(mean_req_throughput)
                aggregate_throughputs.append(aggregate_batch_throughput)
                token_counts.append(total_tokens)

            avg_latency = mean(latencies)
            latency_std = stdev(latencies) if len(latencies) > 1 else 0.0

            avg_mean_throughput = mean(mean_throughputs)
            mean_throughput_std = stdev(mean_throughputs) if len(mean_throughputs) > 1 else 0.0
            
            avg_aggregate_throughput = mean(aggregate_throughputs)
            aggregate_throughput_std = stdev(aggregate_throughputs) if len(aggregate_throughputs) > 1 else 0.0

            avg_tokens = mean(token_counts)

            short_prompt = prompt[:50].replace("\n", " ") + "..." if len(prompt) > 50 else prompt

            print(f"\nPrompt {i}: {short_prompt!r}")
            print(f"  Avg Output Tokens (batch total): {avg_tokens:.1f}")
            print(f"  Avg Latency (batch execution):   {avg_latency:.3f} s ± {latency_std:.3f}")
            print(f"  Mean Decode Throughput:          {avg_mean_throughput:.2f} t/s ± {mean_throughput_std:.2f}")
            print(f"  Aggregate Decode Throughput:     {avg_aggregate_throughput:.2f} t/s ± {aggregate_throughput_std:.2f}")

            list_of_prompt_results.append({
                "prompt_id": prompt_id,
                "topic": topic,
                "prompt": prompt,
                "latencies": latencies,
                "mean_throughputs": mean_throughputs,
                "aggregate_throughputs": aggregate_throughputs,
                "token_counts": token_counts,
                "avg_latency": avg_latency,
                "latency_std": latency_std,
                "avg_mean_throughput": avg_mean_throughput,
                "mean_throughput_std": mean_throughput_std,
                "avg_aggregate_throughput": avg_aggregate_throughput,
                "aggregate_throughput_std": aggregate_throughput_std,
                "avg_tokens": avg_tokens,
            })

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. Saving partial results...")
    finally:
        if list_of_prompt_results:
            try:
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "Unknown GPU"
            except Exception:
                gpu_name = "Unknown GPU"

            results_data = {
                "run_id": run_id,
                "engine": "vllm",
                "config_id": args.config_id,
                "model_name": model_name,
                "spec_method": spec_method,
                "spec_draft_model": spec_draft_model,
                "spec_tokens": spec_tokens,
                "tp_size": num_gpus,
                "gpu_name": gpu_name,
                "bsz": args.bsz,
                "runs_per_prompt": args.runs_per_prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "dataset_file": dataset_file,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "results": list_of_prompt_results
            }

            os.makedirs("results", exist_ok=True)
            filename = f"results/vllm_{args.config_id}_{run_id}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2)
            print(f"Results saved to {filename}")

        print("\nShutting down vLLM server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("Server shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run detailed vLLM OpenAI-server benchmarks."
    )

    parser.add_argument("--config-id", type=str, required=True, choices=list(MODEL_CONFIGS.keys()), help="Model configuration ID to run")
    parser.add_argument("--lite", action="store_true", help="Use lite.json instead of full.json dataset")
    parser.add_argument("--bsz", type=int, default=1, help="Concurrent batch size sent to the server per run")
    parser.add_argument("--runs-per-prompt", type=int, default=8, help="Number of times to run the full batch size for each prompt")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    run_detailed_benchmark(args)