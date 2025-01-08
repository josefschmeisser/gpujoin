#!/bin/bash

declare -r output="hw_comparison_a100.yml"

declare -i key_size=8
declare -i window_size=$((2**22)) # -> 32MiB

# all sizes are specified in terms of the number of tuples
declare -i relation_r_size=$((2**26)) # 0.5GiB
declare -i relation_s_size=$((2**26))
declare -i relation_s_end_size=$((128*1024**3 / key_size)) # 128GiB / key_size
declare -i initial_step=$((128*(10**6))) # -> ~1GiB

function getStep {
    local s_size_gib=$((key_size * relation_s_size / 1024**3))
    #echo $((s_size_gib))
    if [ $s_size_gib -lt 32 ]; then
        echo $((initial_step))
    elif [ $s_size_gib -lt 64 ]; then
        echo $((2 * initial_step))
    else
        echo $((4 * initial_step))
    fi
}

declare -r prefix="numactl --cpunodebind=0"
declare -r cmd="${prefix} ./index_lookup -w ${window_size} --dataset dense -o ${output}"

while [ $relation_s_size -le $relation_s_end_size ]
do
    step=$(getStep)
    echo "current S size: ${relation_s_size}; step: ${step}"

    eval "${cmd} -a hj_warpcore -i no_op -b 512 -l ${relation_r_size} -e ${relation_s_size}"
    eval "${cmd} -a partitioning -i binary_search -l ${relation_r_size} -e ${relation_s_size}"
    eval "${cmd} -a partitioning -i radix_spline -l ${relation_r_size} -e ${relation_s_size}"
    eval "${cmd} -a partitioning -i harmonia -l ${relation_r_size} -e ${relation_s_size}"
    eval "${cmd} -a partitioning -i btree -l ${relation_r_size} -e ${relation_s_size}"

    relation_s_size=relation_s_size+step
done

