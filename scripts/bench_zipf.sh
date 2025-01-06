#!/bin/bash

declare -r output="zipf.yml"

declare -i key_size=8

# all sizes are specified in terms of the number of tuples
declare -i relation_r_size=$((2**26)) # 0.5GiB
declare -i relation_s_size=$((100*1024**3 / key_size)) # 100GiB / key_size

declare -i window_size=$((2**22)) # -> 32MiB

declare -r zipf_step="0.25"
declare -r zipf_parameter_start="0"
declare -r zipf_parameter_end="1.75"
declare zipf_parameter=$zipf_parameter_start

declare -r prefix="numactl --cpunodebind=0"
#declare -r prefix=""
declare -r cmd="${prefix} ./index_lookup -l ${relation_r_size} -e ${relation_s_size} -w ${window_size} --dataset dense -p zipf -o ${output}"

while [ $(bc -l <<< "$zipf_parameter <= $zipf_parameter_end") -eq 1 ]
do
    echo "current zipf parameter: ${zipf_parameter}; step: ${zipf_step}"

    eval "${cmd} -a hj_warpcore -i no_op -z ${zipf_parameter} -b 512"
    eval "${cmd} -a partitioning -i binary_search -z ${zipf_parameter}"
    eval "${cmd} -a partitioning -i radix_spline -z ${zipf_parameter}"
    eval "${cmd} -a partitioning -i harmonia -z ${zipf_parameter}"
    eval "${cmd} -a partitioning -i btree -z ${zipf_parameter}"

    #zipf_parameter=$(echo "$zipf_parameter + $zipf_step" | bc)
    zipf_parameter=$(bc -l <<< "$zipf_parameter + $zipf_step")
done
