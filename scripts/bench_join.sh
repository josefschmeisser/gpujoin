#!/bin/bash

declare -i relation_r_size=$((2**26))
declare -i relation_s_size=0
maxbits=( 26 28 30 32 )
for i in "${maxbits[@]}"
do
    relation_s_size=$((2**i))
    eval "./index_lookup -a plain -i lower_bound -l ${relation_r_size} -e ${relation_s_size} -m 64 --dataset dense"
    eval "./index_lookup -a plain -i radix_spline -l ${relation_r_size} -e ${relation_s_size} -m 64 --dataset dense"
    eval "./index_lookup -a plain -i harmonia -l ${relation_r_size} -e ${relation_s_size} -m 64 --dataset dense"
    eval "./index_lookup -a plain -i btree -l ${relation_r_size} -e ${relation_s_size} -m 64 --dataset dense"
    eval "./index_lookup -a partitioning -i lower_bound -l ${relation_r_size} -e ${relation_s_size} -m 64 --dataset dense"
    eval "./index_lookup -a partitioning -i radix_spline -l ${relation_r_size} -e ${relation_s_size} -m 64 --dataset dense"
    eval "./index_lookup -a partitioning -i harmonia -l ${relation_r_size} -e ${relation_s_size} -m 64 --dataset dense"
    eval "./index_lookup -a partitioning -i btree -l ${relation_r_size} -e ${relation_s_size} -m 64 --dataset dense"
done
