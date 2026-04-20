import subprocess
import time
import atexit
import urllib.request
import urllib.error
from openai import OpenAI

# 1. Define the exact command line arguments as a list to avoid shell parsing/quoting issues
MODEL = "QuantTrio/Qwen3.5-27B-AWQ"
COMMAND = [
    "vllm", "serve", MODEL,
    #"--speculative-config", '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}',
    "--attention-backend", "flash_attn",
    "--max-num-batched-tokens", "8000",
    "--max-model-len", "16000",
    "--gpu-memory-utilization", "0.95"
]

LOG_FILE = "vllm_server.log"
PORT = 8000

def wait_for_server(port=PORT, timeout=600):
    """Polls the vLLM health/models endpoint until it responds."""
    url = f"http://localhost:{port}/v1/models"
    start_time = time.time()
    print(f"Waiting for vLLM server to spin up on port {port}...")
    print(f"(Tailing logs to {LOG_FILE}. This may take a few minutes for VRAM loading...)")
    
    while time.time() - start_time < timeout:
        try:
            # If this succeeds, the server is up and routing traffic
            urllib.request.urlopen(url)
            return True
        except urllib.error.URLError:
            time.sleep(5)
    return False

def main():
    # 2. Launch the server via subprocess
    print("🚀 Launching vLLM subprocess...")
    log_file = open(LOG_FILE, "w")
    server_process = subprocess.Popen(
        COMMAND, 
        stdout=log_file, 
        stderr=subprocess.STDOUT
    )

    # Ensure the subprocess is killed when we exit the python script
    def cleanup():
        print("\n[System] Shutting down vLLM server process...")
        server_process.terminate()
        server_process.wait()
        log_file.close()

    atexit.register(cleanup)

    # 3. Wait for readiness
    if not wait_for_server():
        print("\n❌ Error: Server failed to start within the timeout. Check vllm_server.log for details.")
        return

    print("\n" + "="*50)
    print("✅ Model Engine Ready! You can now chat.")
    print("Type 'exit' or 'quit' to terminate.")
    print("="*50)

    # 4. Initialize the OpenAI-compatible client
    client = OpenAI(
        base_url=f"http://localhost:{PORT}/v1",
        api_key="EMPTY" # vLLM doesn't require an API key by default
    )

    chat_history = []

    # 5. Interactive Chat Loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            chat_history.append({"role": "user", "content": user_input})
            
            print("Assistant: ", end="", flush=True)

            # Metrics tracking
            start_time = time.perf_counter()
            first_token_time = None
            completion_tokens = 0
            full_response = ""

            # Call the local API
            # stream_options={"include_usage": True} forces vLLM to append a final chunk with exact token counts
            response = client.chat.completions.create(
                model=MODEL,
                messages=chat_history,
                temperature=0.7,
                max_tokens=4096*2,
                stream=True,
                stream_options={"include_usage": True} 
            )

            for chunk in response:
                # Track Time-To-First-Token
                if first_token_time is None and chunk.choices:
                    first_token_time = time.perf_counter()

                # Print streamed text
                if chunk.choices and chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    print(text_chunk, end="", flush=True)
                    full_response += text_chunk

                # Grab exact token counts from the final chunk
                if chunk.usage:
                    completion_tokens = chunk.usage.completion_tokens

            end_time = time.perf_counter()
            print() # Print newline after response finishes

            # --- Metrics Calculation ---
            total_time = end_time - start_time
            ttft = first_token_time - start_time if first_token_time else 0
            decode_time = end_time - first_token_time if first_token_time else 0
            tps = completion_tokens / decode_time if decode_time > 0 and completion_tokens > 0 else 0

            print(f"\n[ 📊 Generation Stats ]")
            print(f"  • Completion Tokens : {completion_tokens}")
            print(f"  • TTFT (Prefill)    : {ttft:.3f} s")
            print(f"  • Decode Speed      : {tps:.2f} t/s")
            print(f"  • Total Time        : {total_time:.3f} s")

            # Append to history
            chat_history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            break
        except Exception as e:
            print(f"\n[!] An error occurred: {e}")

if __name__ == "__main__":
    main()