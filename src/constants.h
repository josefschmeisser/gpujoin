#pragma once

#if defined(__aarch64__)
  #define CACHE_LINE_SIZE 64U
#elif defined(__amd64__)
  #define CACHE_LINE_SIZE 64U
#elif defined(__powerpc64__)
  #define CACHE_LINE_SIZE 128U 
#else
  #error Missing CACHE_LINE_SIZE definition for target architecture
#endif

#define GPU_CACHE_LINE_SIZE 128U
#define ALIGN_BYTES 128U
#define LOG2_NUM_BANKS 5U
#define LASWWC_TUPLES_PER_THREAD 5U
