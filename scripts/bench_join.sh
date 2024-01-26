#!/bin/bash

#declare -r output="simple_join_msb.yml"
#declare -r output="simple_join_hj_aggregates_only_50_percent_ht_device_only_2.yml"
declare -r output="simple_join_master_2023-08-07.yml"

declare -i key_size=8
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

while [ $relation_s_size -le $relation_s_end_size ]
do
    step=$(getStep)
    echo "current S size: ${relation_s_size}; step: ${step}"
    eval "numactl --cpunodebind=0 ./index_lookup -a plain -i binary_search -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a plain -i radix_spline -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a plain -i harmonia -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a plain -i btree -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -i binary_search -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -i radix_spline -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -i harmonia -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -i btree -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a hj -i no_op -l ${relation_r_size} -e ${relation_s_size} --dataset dense -p uniform_unique -o ${output}"

    relation_s_size=relation_s_size+step
done
