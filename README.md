# 🖼️ GrayScale | GPU Image Processing Demo
## 1. Introduction

This project demonstrates the conversion of RGB images to Grayscale using **GPU acceleration via OpenCL**. The primary goals include getting familiar with GPU architecture and execution models, and starting with GPU-based parallel processing.

**Environment & Technologies:**

-   GPU: **Intel UHD Graphics 620**
-   API: **OpenCL 3.0**
-   Language: C/C++

----------

## 2. Intel UHD Graphics 620 Architecture
![(intel-gpu-architecture.webp)](https://github.com/ducdmd152/grayscale-opencl/blob/main/intel-gpu-architecture.webp)
-   **24 Execution Units (EUs)**:
    
    -   **3 subslices**, each containing **8 EUs.**
    -   Shared components include:
        -   **L3 Cache**: Shared data cache across all EUs, used to reduce memory latency and improve performance for large datasets.
        -   **L2 Cache**: Each slice (8 EUs) has its own dedicated L2 cache, which helps in reducing access times to frequently used data.
        -   **L1 Cache**: Smaller, faster cache located closer to each individual EU, used for immediate access to data. Helps in improving the speed of thread execution.
        -   **Thread Dispatcher**: Manages and schedules threads.
        -   **Sampler Units**: Handles texture fetch and sampling.
        -   **Shared Resources**: Registers, counters, memory units used across threads.
-   Each **EU** supports up to **7 independent hardware threads**.
    
    👉 Theoretical maximum concurrent threads: `24 EUs × 7 threads = 168 threads`
    
-   Each thread can execute instructions using **SIMD8/SIMD16/SIMD32** (Single Instruction, Multiple Data – up to 8/16/32 data elements per instruction/thread), depending on the compiler's decision based on the complexity of the kernel.
    
    👉 The data processing lanes: ~ 168 threads × 16 = 5376 parallel data lanes.
    
-   No dedicated DRAM — shares system RAM with CPU (Unified Memory).
    
![(No 8, The Compute Architecture of Intel Processor Graphics Gen9)](https://github.com/ducdmd152/grayscale-opencl/blob/main/image.webp)
(No 8, The Compute Architecture of Intel Processor Graphics Gen9)

----------

## 3. Basic Concepts in GPU Programming with OpenCL

### 🔹 OpenCL Environment

-   **Platform**: A vendor (e.g., Intel / AMD / NVIDIA).
-   **Device**: Specific GPU/CPU device.
-   **Context**: A shared workspace between CPU and GPU.
    -   **Buffers**: Used to store input and output image data.
    -   **Program**: Contains the compiled OpenCL kernel code.
    -   **Kernel**: The GPU function executed in parallel.
    -   **Command Queue**: Handles task execution order and sync.

### 🔹 Memory Model

-   **Global Memory**: Accessible by all workgroups.
-   **Local Memory**: Shared within a single workgroup.
-   **Private Memory**: Specific to each work-item.

### 🔹 Execution Model

-   **Work-item**: The smallest unit of execution (similar to a thread).
-   **Work-group**: A group of work-items that run concurrently.
-   The total number of work-items typically equals the number of pixels (one per pixel).

----------

## 4. Solving the Grayscale Problem on GPU

### 🔹 Why GPU?

-   GPUs excel at **massively parallel workloads**—perfect for image processing where each pixel can be processed independently.
-   CPUs are optimized for logic-heavy, control-intensive tasks, while GPUs shine in **data-parallel computation**.

### 🔹 Optimal Configuration:

-   Intel UHD 620: 24 EUs × 7 threads = **168 threads**, assuming each uses SIMD16 = 5376 data lanes.
-   **Work group size = 64** is a practical choice:
    -   Matches typical wavefront/warp sizes (power of 2).
    -   Easy to split image blocks into 8×8 or 16×4 chunks.

### 🔹 Processing Pipeline:

1.  Load image and extract RGB pixel data.
2.  Initialize OpenCL context and select device.
3.  Compile and build kernel to convert RGB → Grayscale.
4.  Enqueue kernel execution.
5.  Read output buffer and save the resulting grayscale image.

### 🔹 RGB to Grayscale Conversion Formula:

```c
gray = 0.299 * R + 0.587 * G + 0.114 * B;
```

### 🔹 **Performance Evaluation:**

-   **GPU (Intel UHD Graphics 620):**
    -   Contains **24 Execution Units (EUs)** × **7 threads per EU**, each running **SIMD16** = **5376 parallel lanes**.
    -   Can process a **1920×1080 image (~2.07 million pixels)** in approximately **15–25 ms**, depending on kernel efficiency.
    -   ⇒ Achieves **~40 to 60 images per second** with optimized pipeline (e.g., asynchronous memory transfer and compute overlap).
-   **CPU (Intel Core i5-8250U @ 1.60–1.80GHz):**
    -   With 4 cores / 8 threads, sequential grayscale conversion takes around **150–300 ms per image**.
    -   ⇒ Processes **~3 to 6 images per second**, **10× slower** than GPU.

⇒ **OpenCL acceleration offers 10–15× speedup over CPU-based processing for grayscale conversion.**

----------

## 5. NVIDIA GPU Comparison
| **Category**                   | **Intel UHD Graphics 620**                                                                 | **NVIDIA RTX 3090**                                                                                     |
|--------------------------------|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Architecture**               | Gen9 (Integrated GPU)                                                                      | Ampere (GA102, Dedicated GPU)                                                                            |
| **Execution Units**            | 24 EUs                                                                                     | 10,496 CUDA Cores                                                                                        |
| **Thread Model**               | 7 threads/EU, SIMD8                                                                         | Warp (32 threads), SIMT                                                                                  |
| **Optimal Workgroup Size**     | 64–128                                                                                     | 128–1024 (multiple of 32, often 256 or 512)                                                              |
| **Memory Architecture**        | Shared system RAM                                                                          | 24 GB GDDR6X, L1/L2 Cache + Shared Memory                                                                |
| **Scheduler Type**             | Static (CPU-side enqueue)                                                                  | Dynamic warp-level scheduling                                                                            |
| **Peak Theoretical Parallelism** | ~1344 SIMD lanes                                                                           | 10,496 cores × 2 (FP32/INT32 dual-issue) = ~20,000+ threads in flight                                                                                                         |
| **Execution Unit Details**     | 24 EUs × 7 threads/EU, SIMD16 = ~5376 parallel lanes                                       | 10,496 CUDA cores × 2 FP32 or INT32 ops = ~20,000 threads                                                |
| **Image Processing Example**   | Processes **1920×1080 image (~2.07 million pixels)** in **15–25 ms** | Achieves **~40 to 60 images per second** with optimized pipeline |

----------

## 6. Live demo

-   Set-up guide: https://github.com/KhronosGroup/OpenCL-Guide?tab=readme-ov-file#the-opencl-sdk
    
-   Command to run (on x64 Native Tools Command Prompt):
    ```bash
    cl.exe /nologo /W4 /DCL_TARGET_OPENCL_VERSION=100 ^
    /I"<PATH_TO_OPENCL_INCLUDE>" <YOUR_SOURCE_FILE.cpp> ^
    /Fe:<OUTPUT_EXE_NAME>.exe /link /LIBPATH:"<PATH_TO_OPENCL_LIB>" OpenCL.lib
    ```
    ```bash
    Example for my device:
    cl.exe /nologo /W4 /DCL_TARGET_OPENCL_VERSION=100 ^
    /ID:\\vcpkg\\vcpkg\\packages\\opencl_x64-windows\\include main.cpp ^
    /Fe:grayscale.exe /link /LIBPATH:D:\\vcpkg\\vcpkg\\packages\\opencl_x64-windows\\lib OpenCL.lib
    ```
    ```bash
    grayscale.exe
    ```
    
-   Live Demo: https://youtu.be/vEB8gr6dpK4.
    

----------

## 7. References

-   📄 [Intel Gen9 GPU Architecture (PDF)](https://cdrdv2-public.intel.com/774710/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0-166010.pdf)
-   📘 [ENCCS GPU Programming Tutorial](https://enccs.github.io/gpu-programming/2-gpu-ecosystem/)
-   📚 [OpenCL Guide – KhronosGroup](https://github.com/KhronosGroup/OpenCL-Guide)
-   📦 [stb_image – Public domain image libs](https://github.com/nothings/stb)
-   📜 [GrayScale - Wikipedia](https://en.wikipedia.org/wiki/Grayscale)

----------

## License & Copyright
&copy; 2025 Duc Dao Licensed under the [MIT LICENSE](https://github.com/ducdmd152/grayscale-opencl/blob/main/LICENSE).

> 🤟 Take a star if you find something interesting 🤟
