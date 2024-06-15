#!/bin/bash

declare -r name="simple_join_tlb_misses_partitioning"

declare -r output="${name}.yml"
declare -r perf_output="${name}_perf.json"

declare -r command_prefix="perf stat --per-socket --event=PM_XTS_ATR_DEMAND_CHECKOUT -x , -o ${perf_output} --append numactl --cpunodebind=0 "

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

eval "echo \"[\" >> ${perf_output}"

while [ $relation_s_size -le $relation_s_end_size ]
do
    step=$(getStep)
    echo "current S size: ${relation_s_size}; step: ${step}"

    #eval "echo -n \"{ \\\"approach\\\": \\\"plain\\\", \\\"index\\\": \\\"binary_search\\\", \\\"num_lookups\\\": ${relation_r_size}, \\\"num_elements\\\": ${relation_s_size}, \\\"perf_output\\\": \\\"\" >> ${perf_output}"
    #eval "${command_prefix} ./index_lookup -a plain -i binary_search -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    #eval "echo \"\\\" },\" >> ${perf_output}"

    #eval "echo -n \"{ \\\"approach\\\": \\\"plain\\\", \\\"index\\\": \\\"radix_spline\\\", \\\"num_lookups\\\": ${relation_r_size}, \\\"num_elements\\\": ${relation_s_size}, \\\"perf_output\\\": \\\"\" >> ${perf_output}"
    #eval "${command_prefix} ./index_lookup -a plain -i radix_spline -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    #eval "echo \"\\\" },\" >> ${perf_output}"

    #eval "echo -n \"{ \\\"approach\\\": \\\"plain\\\", \\\"index\\\": \\\"harmonia\\\", \\\"num_lookups\\\": ${relation_r_size}, \\\"num_elements\\\": ${relation_s_size}, \\\"perf_output\\\": \\\"\" >> ${perf_output}"
    #eval "${command_prefix} ./index_lookup -a plain -i harmonia -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    #eval "echo \"\\\" },\" >> ${perf_output}"

    #eval "echo -n \"{ \\\"approach\\\": \\\"plain\\\", \\\"index\\\": \\\"btree\\\", \\\"num_lookups\\\": ${relation_r_size}, \\\"num_elements\\\": ${relation_s_size}, \\\"perf_output\\\": \\\"\" >> ${perf_output}"
    #eval "${command_prefix} ./index_lookup -a plain -i btree -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    #eval "echo \"\\\" },\" >> ${perf_output}"


    eval "echo -n \"{ \\\"approach\\\": \\\"partitioning\\\", \\\"index\\\": \\\"binary_search\\\", \\\"num_lookups\\\": ${relation_r_size}, \\\"num_elements\\\": ${relation_s_size}, \\\"perf_output\\\": \\\"\" >> ${perf_output}"
    eval "${command_prefix} ./index_lookup -a partitioning -i binary_search -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "echo \"\\\" },\" >> ${perf_output}"

    eval "echo -n \"{ \\\"approach\\\": \\\"partitioning\\\", \\\"index\\\": \\\"radix_spline\\\", \\\"num_lookups\\\": ${relation_r_size}, \\\"num_elements\\\": ${relation_s_size}, \\\"perf_output\\\": \\\"\" >> ${perf_output}"
    eval "${command_prefix} ./index_lookup -a partitioning -i radix_spline -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "echo \"\\\" },\" >> ${perf_output}"

    eval "echo -n \"{ \\\"approach\\\": \\\"partitioning\\\", \\\"index\\\": \\\"harmonia\\\", \\\"num_lookups\\\": ${relation_r_size}, \\\"num_elements\\\": ${relation_s_size}, \\\"perf_output\\\": \\\"\" >> ${perf_output}"
    eval "${command_prefix} ./index_lookup -a partitioning -i harmonia -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "echo \"\\\" },\" >> ${perf_output}"

    eval "echo -n \"{ \\\"approach\\\": \\\"partitioning\\\", \\\"index\\\": \\\"btree\\\", \\\"num_lookups\\\": ${relation_r_size}, \\\"num_elements\\\": ${relation_s_size}, \\\"perf_output\\\": \\\"\" >> ${perf_output}"
    eval "${command_prefix} ./index_lookup -a partitioning -i btree -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "echo \"\\\" },\" >> ${perf_output}"

    #eval "${command_prefix} ./index_lookup -a hj -i no_op -l ${relation_r_size} -e ${relation_s_size} --dataset dense -p uniform_unique -o ${output}"

    relation_s_size=relation_s_size+step
done

eval "echo \"{}]\" >> ${perf_output}"

