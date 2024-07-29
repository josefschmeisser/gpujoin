#!/bin/bash

declare -r name="simple_join_transfer_volume"

declare -r output="${name}.yml"
declare -r nvperf_output="${name}_nvperf.json"
declare -r nvperf_log=$(mktemp --suffix ".csv")

declare -r command_prefix="/usr/local/cuda-11.1/bin/nvprof --csv --log-file ${nvperf_log} --replay-mode kernel"\
"--aggregate-mode on --metrics nvlink_total_data_received,nvlink_user_data_received --"\
"numactl --cpunodebind=0 "

declare -i key_size=8
# TODO adapt
declare -i relation_r_size=$((1*1024**3 / key_size))
declare -i relation_s_size=$((1*1024**3 / key_size))
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

function createCommand {
    # TODO distribution
    printf '%s' \
        "${command_prefix}" \
        "./index_lookup -a $1 -i $2 -l ${relation_r_size}" \
        " -e ${relation_s_size} --d dense -p uniform_unique -o ${output}"
}

function startEntry {
    local json=$(printf '%s' \
        "{ \"approach\": \"$1\", \"index\": \"$2\", \"num_lookups\": ${relation_r_size}, " \
        "\"num_elements\": ${relation_s_size}, \"nvperf_output\": \""
    )
    echo -n ${json} >> ${nvperf_output}
    #eval "echo -n ${json}"
}

function finalizeEntry {
    sed 's/^=.*//' ${nvperf_log}
    sed 's/"//' ${nvperf_log}
    cat ${nvperf_log} >> ${nvperf_output}
    echo "\" }," >> ${nvperf_output}
    echo -n > ${nvperf_log}
}

function runApproach {
    startEntry $1 $2
    eval $(createCommand $1 $2)
    finalizeEntry
}

eval "echo \"[\" >> ${nvperf_output}"

while [ $relation_s_size -le $relation_s_end_size ]
do
    step=$(getStep)
    echo "current S size: ${relation_s_size}; step: ${step}"

    runApproach "plain" "binary_search"
    runApproach "plain" "radix_spline"
    runApproach "plain" "harmonia"
    runApproach "plain" "btree"
    runApproach "hj" "no_op"

    relation_s_size=relation_s_size+step
done

eval "echo \"{}]\" >> ${nvperf_output}"
