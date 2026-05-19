import time
import torch
import argparse
from statistics import mean, stdev
from vllm import LLM, SamplingParams
from tqdm import tqdm

def run_detailed_benchmark(args):
    model_registry = {
        "9b": {
            "base": "QuantTrio/Qwen3.5-9B-AWQ",
            "spec": "z-lab/Qwen3.5-9B-DFlash"
        },
        "27b": {
            "base": "QuantTrio/Qwen3.5-27B-AWQ",
            "spec": "z-lab/Qwen3.5-27B-DFlash"
        },
        "35b3": {
            "base": "QuantTrio/Qwen3.5-35B-A3B-AWQ",
            "spec": "z-lab/Qwen3.5-35B-A3B-DFlash"
        }
    }

    model_name = model_registry[args.model_size]["base"]
    spec_model_name = model_registry[args.model_size]["spec"]

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")

    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": num_gpus,
        "max_model_len": 16000,
        "gpu_memory_utilization": 0.9,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 1,
        "disable_custom_all_reduce": True,
        "enable_prefix_caching": True,
    }

    if args.use_spec:
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": spec_model_name,
            "num_speculative_tokens": 15,
        }

    print(f"Loading Base Model: {model_name}")
    if args.use_spec:
        print(f"Speculative Decoding Enabled with: {spec_model_name}")

    llm = LLM(**llm_kwargs)

    prompts = [
        "Let PP be a polygon formed by the edges of an infinite chessboard, which does not intersect itself. Let the numbers a1,a2,a3a1​,a2​,a3​ represent the number of unit squares that have exactly 1,2 or 31,2 or 3 edges on the boundary of PP respectively. Find the largest real number kk such that the inequality a1+a2>ka3a1​+a2​>ka3​ holds for each polygon constructed with these conditions.",
        "Let f:R→Rf:R→R be a continuous function. A chord is defined as a segment of integer length, parallel to the x-axis, whose endpoints lie on the graph of ff. It is known that the graph of ff contains exactly NN chords, one of which has length 2025. Find the minimum possible value of NN. ",
        "We call an n×nn×n table filled with positive integers *divisoral* if it holds that: numbers in ii-th row are exactly all divisors of some positive integer riri​, numbers in jj-th column are exactly all divisors of some positive integer cjcj​, and ri≠rjri​=rj​ for each i≠ji=j. We are given a prime pp. Let S(p)S(p) be the smallest positive integer nn, divisible by pp, such that there exists a divisoral table of size n×nn×n. Find the sum of S(p)S(p) for all primes p≤13p≤13.",
        "Hannah has a 2024×20252024×2025 rectangle in the coordinate plane, with sides parallel to the axes. She makes a cut from one side to another side, which only goes down and/or right along grid lines. Then she puts the two pieces together, possibly with rotations and/or reflections without overlaps or gaps, to form a new rectangle which is not congruent to the original. How many possible new rectangles can she produce? (An a×ba×b rectangle is considered the same as a b×ab×a rectangle.)",
        "Hummingbirds within Apodiformes uniquely have a bilaterally paired oval bone, a sesamoid embedded in the caudolateral portion of the expanded, cruciate aponeurosis of insertion of m. depressor caudae. How many paired tendons are supported by this sesamoid bone? Answer with a number.",
        "In Greek mythology, who was Jason's maternal great-grandfather?",
        """I am providing the standardized Biblical Hebrew source text from the Biblia Hebraica Stuttgartensia (Psalms 104:7). Your task is to distinguish between closed and open syllables. Please identify and list all closed syllables (ending in a consonant sound) based on the latest research on the Tiberian pronunciation tradition of Biblical Hebrew by scholars such as Geoffrey Khan, Aaron D. Hornkohl, Kim Phillips, and Benjamin Suchard. Medieval sources, such as the Karaite transcription manuscripts, have enabled modern researchers to better understand specific aspects of Biblical Hebrew pronunciation in the Tiberian tradition, including the qualities and functions of the shewa and which letters were pronounced as consonants at the ends of syllables.\n\nמִן־גַּעֲרָ֣תְךָ֣ יְנוּס֑וּן מִן־ק֥וֹל רַֽ֝עַמְךָ֗ יֵחָפֵזֽוּן (Psalms 104:7) ?"""
    ]
    
    tokenizer = llm.get_tokenizer()

    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    warmup_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
    )

    print("Warmup, precache prompts...")
    llm.generate(formatted_prompts, warmup_params, use_tqdm=True)

    print("\n" + "=" * 80)
    print(f"Benchmark Running... (BSZ=1, TEMP={args.temperature}, TOKENS={args.max_tokens}, GENERATIONS={args.runs_per_prompt}, MODEL SIZE={args.model_size}, TP={num_gpus})")
    print("=" * 80)

    for i, prompt in tqdm(enumerate(formatted_prompts, 1)):
        latencies = []
        e2e_speeds = []
        token_counts = []

        for _ in tqdm(range(args.runs_per_prompt), leave=False):
            start = time.time()
            output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
            total_time = time.time() - start

            num_output_tokens = len(output.outputs[0].token_ids)

            e2e_speed = num_output_tokens / total_time

            latencies.append(total_time)
            e2e_speeds.append(e2e_speed)
            token_counts.append(num_output_tokens)

        avg_latency = mean(latencies)
        latency_std = stdev(latencies) if len(latencies) > 1 else 0

        avg_speed = mean(e2e_speeds)
        speed_std = stdev(e2e_speeds) if len(e2e_speeds) > 1 else 0

        avg_tokens = mean(token_counts)

        short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt

        print(f"\nPrompt {i}: {short_prompt!r}")
        print(f"  Avg Output Tokens:        {avg_tokens:.1f}")
        print(f"  Avg Latency:              {avg_latency:.3f} s ± {latency_std:.3f}")
        print(f"  Cached Decode Throughput: {avg_speed:.2f} t/s ± {speed_std:.2f}")
        print(f"  Min/Max Tokens:           {min(token_counts)} / {max(token_counts)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detailed vLLM benchmarks with dynamic model routing.")
    
    parser.add_argument("--model-size", type=str, required=True, choices=["9b", "27b", "35b3"],
                        help="Select the model size. Maps to Qwen3.5 AWQ variants.")
    parser.add_argument("--use-spec", action="store_true",
                        help="Enable speculative decoding (DFlash).")
    parser.add_argument("--runs-per-prompt", type=int, default=8,
                        help="Number of times to benchmark each prompt. (Default: 8)")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Maximum tokens to generate per prompt. (Default: 2000)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for token sampling. (Default: 1.0)")

    args = parser.parse_args()
    
    run_detailed_benchmark(args)