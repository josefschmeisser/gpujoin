#!/bin/bash

declare -r output="2024-06-04_window_size.yml"

declare -i key_size=8
# all sizes in terms of the number of tuples
declare -i relation_r_size=$((2**26)) # 0.5GiB
declare -i relation_s_size=$((100*1024**3 / key_size)) # 100GiB / key_size
#declare -i window_start_size=$((2**22)) # -> 32MiB
declare -i window_start_size=$((2**18)) # -> 2MiB
declare -i window_size=window_start_size
#declare -i step=window_start_size
declare -i initial_step=window_start_size

function getStep {
    if [ $window_size -lt $((2**22)) ]; then
        echo $((initial_step))
    else
        echo $((2**22))
    fi
}

while [ $window_size -le $relation_r_size ]
do
    step=$(getStep)

    echo "current window size: ${window_size}; step: ${step}"

    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -w ${window_size} -i binary_search -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -w ${window_size} -i radix_spline -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -w ${window_size} -i harmonia -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -w ${window_size} -i btree -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"

    window_size=window_size+step
done

