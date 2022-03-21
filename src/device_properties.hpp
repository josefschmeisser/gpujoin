#pragma once

#include <cuda_runtime_api.h>

const cudaDeviceProp& get_device_properties(unsigned device_id);
