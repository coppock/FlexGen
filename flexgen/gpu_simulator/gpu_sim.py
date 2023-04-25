import time


GPU_MEM_FILE_PATH = "/tmp/745-mem-usage"


def get_current_gpu_mem():
    with open(GPU_MEM_FILE_PATH, "r") as gpu_mem_file:
        for memory in gpu_mem_file:
            if memory.strip() != "":
                return int(memory.strip())


def simulate_gpu_memory_utilization(trace_file_path):
    trace = []

    with open(trace_file_path, "r") as trace_file:
        for line in trace_file:
            tokens = line.strip().split(" ")
            start, end, usage = int(tokens[0]), int(tokens[1]), int(tokens[2])
            trace.append((start, end, usage))
    
    trace.sort()

    current_second = 0
    current_GPU_memory_utilization = 0
    working_p = []

    while len(trace) > 0 or len(working_p) > 0:

        before = time.time() / 1000

        while len(trace) > 0 and trace[0][0] == current_second:
            current_GPU_memory_utilization += trace[0][2]
            working_p.append((trace[0][1], trace[0][2]))
            trace = trace[1:]

        working_p.sort()
        
        while len(working_p) > 0 and working_p[0][0] == current_second:
            current_GPU_memory_utilization -= working_p[0][1]
            working_p = working_p[1:]
        
        current_second += 1

        with open(GPU_MEM_FILE_PATH, "w") as gpu_memory_file:
            gpu_memory_file.write(str(current_GPU_memory_utilization))

        after = time.time() / 1000
        time_difference = after - before
        time.sleep(1.0 - time_difference)

        print(current_second, current_GPU_memory_utilization)


if __name__ == "__main__":
    simulate_gpu_memory_utilization("traces/sample.trace")