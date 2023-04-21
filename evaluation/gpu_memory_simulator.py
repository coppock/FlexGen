import sys

gpu_memory_file = open("/opt/local/gpumemsim", "w")

def simulate_gpu_memory(processes_trace):
    print(processes_trace)
    pass

def parse_trace_file(trace_file):
    processes_trace = []

    for line in trace_file:
        # <start_second> <end_second> <memory_usage_in_bytes>
        start, end, usage = line.split(" ")
        processes_trace.append((int(start), int(end), int(usage)))

    sorted(processes_trace, key=lambda x: x[0])

if __name__ == "__main__":
    trace_file = open(sys.argv[1])
    processes_trace = parse_trace_file(trace_file)
    simulate_gpu_memory(processes_trace)