#!/bin/bash

declare -i relation_r_size=$((2**26))
declare -i relation_s_size=$((2**26))
declare -i relation_s_end_size=$((2**34))
declare -i step=$((128*(10**6)))

while [ $relation_s_size -le $relation_s_end_size ]
do
    echo "current S size: ${relation_s_size}"
    eval "./index_lookup -a plain -i binary_search -l ${relation_r_size} -e ${relation_s_size} --dataset dense"
    eval "./index_lookup -a plain -i radix_spline -l ${relation_r_size} -e ${relation_s_size} --dataset dense"
    eval "./index_lookup -a plain -i harmonia -l ${relation_r_size} -e ${relation_s_size} --dataset dense"
    eval "./index_lookup -a plain -i btree -l ${relation_r_size} -e ${relation_s_size} --dataset dense"
    eval "./index_lookup -a partitioning -i binary_search -l ${relation_r_size} -e ${relation_s_size} --dataset dense"
    eval "./index_lookup -a partitioning -i radix_spline -l ${relation_r_size} -e ${relation_s_size} --dataset dense"
    eval "./index_lookup -a partitioning -i harmonia -l ${relation_r_size} -e ${relation_s_size} --dataset dense"
    eval "./index_lookup -a partitioning -i btree -l ${relation_r_size} -e ${relation_s_size} --dataset dense"
    relation_s_size=relation_s_size+step
done
