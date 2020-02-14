#!/usr/bin/env bash
for((i=1;i<=10;i++));
do
python random_test.py --seed $(expr $i \* 100 + 10000) &
done
