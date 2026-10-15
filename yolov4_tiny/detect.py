
import cv2
import time
import numpy as np
import psutil

# Load YOLOv4-Tiny model
net = cv2.dnn.readNetFromDarknet(
    "yolov4-tiny.cfg",
    "yolov4-tiny.weights"
)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("YOLOv4-Tiny loaded successfully!")

# Load test image
image = cv2.imread("test.jpg")

if image is None:
    raise FileNotFoundError("test.jpg not found")

# Preprocess image
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1/255.0,
    size=(416, 416),
    swapRB=True,
    crop=False
)

net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()

# Warm-up
print("\nWarming up...")
for _ in range(2):
    net.forward(output_layers)

# Benchmark configuration
num_runs = 10
latencies = []
cpu_usages = []
memory_usages = []

process = psutil.Process()

print("\nRunning YOLOv4-Tiny inference 10 times...\n")

for i in range(num_runs):

    # CPU time before inference
    cpu_before = process.cpu_times()
    wall_start = time.perf_counter()

    # Run inference
    outputs = net.forward(output_layers)

    wall_elapsed = time.perf_counter() - wall_start
    cpu_after = process.cpu_times()

    # Calculate CPU usage
    cpu_elapsed = (
        cpu_after.user + cpu_after.system
        - cpu_before.user - cpu_before.system
    )

    cpu_usage = 100 * cpu_elapsed / wall_elapsed

    # Process memory usage
    memory_mb = process.memory_info().rss / (1024 * 1024)

    latencies.append(wall_elapsed)
    cpu_usages.append(cpu_usage)
    memory_usages.append(memory_mb)

    print(
        f"Run {i+1:2d}: "
        f"Latency = {wall_elapsed:.3f} s, "
        f"FPS = {1/wall_elapsed:.2f}, "
        f"CPU = {cpu_usage:.1f}%, "
        f"RAM = {memory_mb:.1f} MB"
    )

# Calculate averages
avg_latency = np.mean(latencies)
avg_fps = 1 / avg_latency
avg_cpu = np.mean(cpu_usages)
avg_ram = np.mean(memory_usages)

print("\n========== BENCHMARK RESULTS ==========")
print(f"Number of runs:       {num_runs}")
print(f"Average latency:      {avg_latency:.3f} seconds")
print(f"Average latency:      {avg_latency*1000:.2f} ms")
print(f"Average FPS:          {avg_fps:.2f}")
print(f"Average CPU usage:    {avg_cpu:.1f}%")
print(f"Average RAM usage:    {avg_ram:.1f} MB")
print(f"Maximum sampled RAM:  {max(memory_usages):.1f} MB")
print(f"Output tensors:       {[o.shape for o in outputs]}")
print("=======================================")
