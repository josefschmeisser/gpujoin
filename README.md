This work is part of a research project that investigates index structure
accesses over fast interconnects. We thereby assume that index structures are
stored out-of-core and accessible through interconnect technologies like NVLink
2.0.

The project itself consists of a series of carefully designed
microbenchmarks. Each of the benchmarks aims to highlight or respectively solve
the challenges imposed by the data transfer bottleneck.

## Requirements
- [CUDA-capable device](https://developer.nvidia.com/cuda-gpus) with device architecture 7.0 or higher
- [NVIDIA CUDA toolkit/compiler](https://developer.nvidia.com/cuda-toolkit) version $\ge$ v11.2
- C++14 or higher

## Building
```
mkdir -p build
cd build
cmake -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_70,code=sm_70" ..
make
```

The `gencode` argument should be adapted accordingly to reflect the target GPU's
capabilities.

## Benchmarking
After the build process succeeded all binaries required by the benchmark scripts
can be found in the previously created build directory. The benchmark scripts
themselves are located in `scripts`.

In order to reproduce our results, it is sufficient to execute the corresponding
benchmark script. If, for example, we want to recreate throughput results, we
would run the following from within the build directory:

```
../bench_join_throughput.sh
```

This will generate a `.yml` file in the current directory. It contains
information like the execution time, and parameters such as the used memory
allocators.

**Note:** Certain parameters may require adjustment in order to reflect the
target machine's environment.

## Research
We plan to publish our findings from this project.
