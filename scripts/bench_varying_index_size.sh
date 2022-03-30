#!/bin/bash

#lookup_count=$((2**10))
declare -i index_size=0
maxbits=( 12 14 16 18 20 22 24 )
for i in "${maxbits[@]}"
do
    index_size=$((2 ** (i - 1)))
    echo "run with ${i} max bits and ${index_size} elements"
    eval "./index_lookup --maxbits ${i} --elements ${index_size}"
    eval "./index_lookup -p --maxbits ${i} --elements ${index_size}"
    eval "./async --maxbits ${i} --elements ${index_size}"
done
