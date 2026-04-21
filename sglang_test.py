import argparse
import json
import subprocess
import sys
import time
import threading
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
from tqdm import tqdm

def stream_logs(pipe):
    """Continuously print server logs."""
    for line in pipe:
        sys.stdout.write("[SGLANG] " + line)
    pipe.close()

def launch_sglang_server(model_name, spec_model_name, tp_size, use_spec, port):
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--port", str(port),
        "--tp-size", str(tp_size),
        "--dtype", "bfloat16",
        "--mem-fraction-static", "0.95",
        "--context-length", "4000",
        "--log-level", "error",
        "--mamba-scheduler-strategy", "extra_buffer",
        "--max-prefill-tokens", "2000",
        "--trust-remote-code",
    ]

    if use_spec:
        cmd += [
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", spec_model_name,
            "--speculative-num-draft-tokens", "16",
        ]

    print("\nLaunching SGLang server...")
    print(" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    threading.Thread(target=stream_logs, args=(proc.stdout,), daemon=True).start()

    # wait for server to be ready
    health_url = f"http://localhost:{port}/v1/models"
    print("Waiting for SGLang server to start (this may take a few minutes)...")
    
    for _ in range(600):
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print("SGLang server is ready.\n")
                return proc
        except Exception:
            pass
        time.sleep(2)

    proc.terminate()
    raise RuntimeError("SGLang server failed to start within the timeout.")

def generate_sglang(port, prompt, temperature, max_tokens, verbose=False):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_tokens
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
        print("full_choice:", json.dumps(choice, ensure_ascii=False))

    text = msg.get("content") or ""
    tokens = data["usage"]["completion_tokens"]
    return text, tokens

def run_detailed_benchmark(args):
    model_registry = {
        "9b": {
            "base": "Qwen/Qwen3.5-9B",
            "awq": "QuantTrio/Qwen3.5-9B-AWQ",
            "spec": "z-lab/Qwen3.5-9B-DFlash",
        },
        "27b": {
            "base": "Qwen/Qwen3.5-27B",
            "awq": "QuantTrio/Qwen3.5-27B-AWQ",
            "spec": "z-lab/Qwen3.5-27B-DFlash",
        },
        "35b3": {
            "base": "Qwen/Qwen3.5-35B-A3B",
            "awq": "QuantTrio/Qwen3.5-35B-A3B-AWQ",
            "spec": "z-lab/Qwen3.5-35B-A3B-DFlash",
        },
    }

    if args.awq:
        model_name = model_registry[args.model_size]["awq"]
    else:
        model_name = model_registry[args.model_size]["base"]
        
    spec_model_name = model_registry[args.model_size]["spec"]

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")

    print(f"Target Model: {model_name} (AWQ: {args.awq})")
    if args.use_spec:
        print(f"Speculative Model: {spec_model_name}")

    dataset_file = "data/lite.json" if args.lite else "data/full.json"
    print(f"Loading dataset from: {dataset_file}")
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        prompts = [item["prompt"] for item in dataset]
    except FileNotFoundError:
        print(f"Error: {dataset_file} not found. Ensure the dataset exists.")
        sys.exit(1)

    server_proc = launch_sglang_server(
        model_name=model_name,
        spec_model_name=spec_model_name,
        tp_size=num_gpus,
        use_spec=args.use_spec,
        port=args.port,
    )

    try:
        print("Warmup...")
        for p in prompts[:1]:
            generate_sglang(args.port, p, temperature=0.0, max_tokens=32)

        print("\n" + "=" * 80)
        print(
            f"Benchmark Running... (BSZ={args.bsz}, TEMP={args.temperature}, "
            f"TOKENS={args.max_tokens}, RUNS={args.runs_per_prompt}, "
            f"MODEL={args.model_size}, TP={num_gpus})"
        )
        print("=" * 80)

        for i, prompt in tqdm(enumerate(prompts, 1), total=len(prompts)):
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
                    _, num_tokens = generate_sglang(
                        args.port, 
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

            short_prompt = prompt[:50].replace('\n', ' ') + "..." if len(prompt) > 50 else prompt

            print(f"\nPrompt {i}: {short_prompt!r}")
            print(f"  Avg Output Tokens (batch total): {avg_tokens:.1f}")
            print(f"  Avg Latency (batch execution):   {avg_latency:.3f} s ± {latency_std:.3f}")
            print(f"  Mean Decode Throughput:          {avg_mean_throughput:.2f} t/s ± {mean_throughput_std:.2f}")
            print(f"  Aggregate Decode Throughput:     {avg_aggregate_throughput:.2f} t/s ± {aggregate_throughput_std:.2f}")

    finally:
        print("\nShutting down SGLang server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("Server shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run detailed SGLang OpenAI-server benchmarks."
    )

    parser.add_argument("--model-size", type=str, required=True, choices=["9b", "27b", "35b3"])
    parser.add_argument("--awq", action="store_true", help="Use the AWQ version of the selected model")
    parser.add_argument("--use-spec", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--lite", action="store_true", help="Use lite.json instead of full.json dataset")
    parser.add_argument("--bsz", type=int, default=1, help="Concurrent batch size sent to the server per run")
    parser.add_argument("--runs-per-prompt", type=int, default=8, help="Number of times to run the full batch size for each prompt")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--port", type=int, default=30000)

    args = parser.parse_args()
    run_detailed_benchmark(args)