#include <iostream>




    void compare_join_results(join_entry* ref, unsigned ref_size, join_entry* actual, unsigned actual_size) {
        std::unordered_map<uint32_t, uint32_t> map;
        for (unsigned i = 0; i < ref_size; ++i) {
            if (map.count(ref[i].lineitem_tid) > 0) {
                std::cerr << "lineitem tid " << ref[i].lineitem_tid << " already in map" << std::endl;
                exit(0);
            }
            map.emplace(ref[i].lineitem_tid, ref[i].part_tid);
        }
        for (unsigned i = 0; i < actual_size; ++i) {
            auto it = map.find(actual[i].lineitem_tid);
            if (it != map.end()) {
                if (it->second != actual[i].part_tid) {
                    std::cerr << "part tid " << actual[i].part_tid << " expected " << it->second << std::endl;
                }
            } else {
                std::cerr << "lineitem tid " << actual[i].lineitem_tid << " not in reference" << std::endl;
            }
        }
    }

    void run_two_phase_ij_buffer_debug() {
        decltype(output_index) matches1 = 0;
        decltype(output_index) matches2 = 0;
        decltype(output_index) zero = 0;

        enum { BLOCK_THREADS = 128, ITEMS_PER_THREAD = 8 }; // TODO optimize

        join_entry* join_entries1;
        cudaMallocManaged(&join_entries1, sizeof(join_entry)*lineitem_size);

        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int num_blocks = num_sms*2; // TODO

        const auto start1 = std::chrono::high_resolution_clock::now();
        ij_lookup_kernel_3<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        cudaDeviceSynchronize();

        cudaError_t error = cudaMemcpyFromSymbol(&matches1, output_index, sizeof(matches1), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        printf("join matches1: %u\n", matches1);
        printf("debug_cnt: %u\n", debug_cnt);

        error = cudaMemcpyToSymbol(output_index, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);
        assert(error == cudaSuccess);
        join_entry* join_entries2;
        cudaMallocManaged(&join_entries2, sizeof(join_entry)*lineitem_size);
        num_blocks = (part_size + block_size - 1) / block_size;
        ij_lookup_kernel<<<num_blocks, block_size>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries2);
        cudaDeviceSynchronize();

        error = cudaMemcpyFromSymbol(&matches2, output_index, sizeof(matches2), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        printf("join matches2: %u\n", matches2);

        compare_join_results(join_entries2, matches2, join_entries1, matches1);
        compare_join_results(join_entries1, matches1, join_entries2, matches2);

        const auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start1).count()/1000.;
        std::cout << "kernel time: " << d1 << " ms\n";

        num_blocks = (lineitem_size + block_size - 1) / block_size;

        const auto start2 = std::chrono::high_resolution_clock::now();
        ij_join_kernel<<<num_blocks, block_size>>>(lineitem_device, part_device, join_entries1, matches1);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - start2).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
    }

    void run_two_phase_ij_buffer() {
        using namespace std;

        decltype(output_index) matches1 = 0;

        enum { BLOCK_THREADS = 256, ITEMS_PER_THREAD = 10 }; // TODO optimize

        join_entry* join_entries1;
        cudaMalloc(&join_entries1, sizeof(join_entry)*lineitem_size);

        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int num_blocks = num_sms*4; // TODO

        const auto start1 = std::chrono::high_resolution_clock::now();
        //ij_lookup_kernel<<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        //ij_lookup_kernel_2<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        ij_lookup_kernel_3<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        //ij_lookup_kernel_4<BLOCK_THREADS><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        cudaDeviceSynchronize();
        const auto d1 = chrono::duration_cast<chrono::microseconds>(std::chrono::high_resolution_clock::now() - start1).count()/1000.;
        std::cout << "kernel time: " << d1 << " ms\n";

        cudaError_t error = cudaMemcpyFromSymbol(&matches1, output_index, sizeof(matches1), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        printf("join matches1: %u\n", matches1);

        num_blocks = (lineitem_size + block_size - 1) / block_size;

        const auto start2 = std::chrono::high_resolution_clock::now();
        ij_join_kernel<<<num_blocks, block_size>>>(lineitem_device, part_device, join_entries1, matches1);
        cudaDeviceSynchronize();
        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - start2).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
        std::cout << "complete time: " << d1 + kernelTime << " ms\n";
    }
};

template<class IndexType>
void load_and_run_ij(const std::string& path, bool as_full_pipline_breaker) {
    if (prefetch_index) { throw "not implemented"; }

    helper<IndexType> h;
    h.load_database(path);
    if (as_full_pipline_breaker) {
        printf("full pipline breaker\n");
        h.run_two_phase_ij_buffer();
    } else {
        h.run_ij();
        //h.run_ij_buffer();
    }
}

int main(int argc, char** argv) {
    using namespace std;

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);// devId);

#ifdef USE_HJ
    if (argc != 2) {
        printf("%s <tpch dataset path>\n", argv[0]);
        return 0;
    }

    helper<int> h;
    h.load_database(argv[1]);
    //std::getc(stdin);
    h.run_hj();
#else
    if (argc < 3) {
        printf("%s <tpch dataset path> <index type: {0: btree, 1: harmonia, 2: radix_spline, 3: binary_search> <1: full pipline breaker>\n", argv[0]);
        return 0;
    }
    enum IndexType : unsigned { btree, harmonia, radix_spline, binary_search, nop } index_type { static_cast<IndexType>(std::stoi(argv[2])) };
    bool full_pipline_breaker = (argc < 4) ? false : std::stoi(argv[3]) != 0;

#ifdef SKIP_SORT
    std::cout << "skip sort step: yes" << std::endl;
#else
    std::cout << "skip sort step: no" << std::endl;
#endif

    switch (index_type) {
        case IndexType::btree: {
            printf("using btree\n");
            using index_type = btree_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        case IndexType::harmonia: {
            printf("using harmonia\n");
            using index_type = harmonia_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        case IndexType::radix_spline: {
            printf("using radix_spline\n");
            using index_type = radix_spline_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        case IndexType::binary_search: {
            printf("using binary search\n");
            using index_type = binary_search_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        case IndexType::nop: {
            printf("using no_op_index\n");
            using index_type = no_op_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        default:
            std::cerr << "unknown index type: " << index_type << std::endl;
            return 0;
    }
#endif

/*
    printf("sum1: %lu\n", globalSum1);
    printf("sum2: %lu\n", globalSum2);
*/
    const int64_t result = 100*(globalSum1*1'000)/(globalSum2/1'000);
    printf("%ld.%ld\n", result/1'000'000, result%1'000'000);

    std::cout << std::setprecision(2) << std::scientific
        << "scan_cycles: " << (double)scan_cycles
        << "; sync_cycles: " << (double)sync_cycles
        << "; sort_cycles: " << (double)sort_cycles
        << "; lookup_cycles: " << (double)lookup_cycles
        << "; join_cycles: " << (double)join_cycles
        << "; total_cycles: " << (double)total_cycles
        << std::endl; 

    cudaDeviceReset();

    return 0;
}
