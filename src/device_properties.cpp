#include "device_properties.hpp"

#include <memory>
#include <unordered_map>

#include <cub/util_debug.cuh>
#undef _Float16

#include "utils.hpp"

struct device_properties_cache {
    std::unordered_map<unsigned, std::unique_ptr<cudaDeviceProp>> cache;

    const cudaDeviceProp& get(unsigned device_id) {
        if (cache.count(device_id) > 0) {
            return *cache[device_id];
        }

        // fetch device properties for the given device
        auto entry = std::make_unique<cudaDeviceProp>();
        CubDebugExit(cudaGetDeviceProperties(entry.get(), device_id));
        cache[device_id] = std::move(entry);

        return *cache[device_id];
    }
};

const cudaDeviceProp& get_device_properties(unsigned device_id) {
    return singleton<device_properties_cache>::instance().get(device_id);
}
