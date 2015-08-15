#!/bin/bash
for ((i=6; i<13; i+=1)); do
  let mnk=(2**$i)
  echo "BLAS-client -o1 --transposeB 1 -m $mnk -n $mnk -k $mnk" >> bench.txt
  clBLAS-client -o1 --transposeB 1 -m $mnk -n $mnk -k $mnk >> bench.txt
done

for ((i=1; i<13; i+=1)); do
  let mnk=(512*$i)
  echo "BLAS-client -o1 --transposeB 1 -m $mnk -n $mnk -k $mnk" >> bench.txt
  clBLAS-client -o1 --transposeB 1 -m $mnk -n $mnk -k $mnk >> bench.txt
done

