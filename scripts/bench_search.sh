#!/bin/bash

declare -i relation_r_size=$((2**26))
declare -i relation_s_size=$((2**26))
declare -i relation_s_end_size=$((2**30))
declare -i step=$((128*(10**6)))
declare -r output="search.yml"

while [ $relation_s_size -le $relation_s_end_size ]
do
    echo "current S size: ${relation_s_size}"
    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -i binary_search -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    eval "numactl --cpunodebind=0 ./index_lookup -a partitioning -i radix_spline -l ${relation_r_size} -e ${relation_s_size} --dataset dense -o ${output}"
    relation_s_size=relation_s_size+step
done
