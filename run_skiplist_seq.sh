#!/bin/bash

make

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi


declare -a num_ops=("10000" "50000" "100000", "500000", "1000000")
declare -a key_ranges=("100" "1000" "10000" "100000")

for b in "${key_ranges[@]}"; do
    for a in "${num_ops[@]}"; do
        time=$(./test 20 20 $a $b | grep -o -E '[0-9]+\.[0-9]+')

        echo "Time for config ./test 20 20 $a $b: $time ms" 
    done
done

for b in "${key_ranges[@]}"; do
    for a in "${num_ops[@]}"; do
        time=$(./test 40 40 $a $b | grep -o -E '[0-9]+\.[0-9]+')

        echo "Time for config ./test 40 40 $a $b: $time ms" 
    done
done


