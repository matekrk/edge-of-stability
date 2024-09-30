#!/bin/bash

# Define your arguments
args1=("0" "1" "2")
args2=("0.001")
args3=("20" "50" "100")

# Get the length of the arrays
length1=${#args1[@]}
length2=${#args2[@]}
length3=${#args3[@]}

# Use a for loop to iterate over the arrays
for ((i=0; i<$length1; i++)); do
  for ((j=0; i<$length2; i++)); do
    for ((k=0; i<$length3; i++)); do
      nohup python sp.py ${args1[$i]} ${args2[$j]} ${args3[$k]} > outputs/sp_${i}_${j}_${k}.sh &
    done
  done
done
