import subprocess
import time
import psutil
import os

def run_and_monitor(script_path):
    print(f"Running {script_path}...")
    start_time = time.time()

    # Start the process
    process = subprocess.Popen(["python", script_path])
    proc = psutil.Process(process.pid)

    peak_memory = 0

    # Monitor while process is running
    while process.poll() is None:
        try:
            mem = proc.memory_info().rss / (1024 * 1024)  # in MB
            peak_memory = max(peak_memory, mem)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)

    end_time = time.time()
    duration = end_time - start_time

    return {
        "script": script_path,
        "execution_time_sec": round(duration, 2),
        "peak_memory_MB": round(peak_memory, 2)
    }

# Run both scripts
results = []
for script in ["src/dataPrepPandas.py", "src/dataPrepPolars.py"]:
    results.append(run_and_monitor(script))

# Save results
with open("benchmark_results.txt", "w") as f:
    for result in results:
        f.write(f"{result['script']}:\n")
        f.write(f"  Execution Time: {result['execution_time_sec']} seconds\n")
        f.write(f"  Peak Memory Usage: {result['peak_memory_MB']} MB\n\n")

print("Benchmarking complete. Results saved to benchmark_results.txt.")
