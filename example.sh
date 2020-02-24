#!/usr/bin/env bash

# train model
python DQN.py --device cpu --double-dqn --lr 0.001 --package-num 20  --func-type distance/uniform/constant --time-limit 2  --num-env 32 --max-step 20000

# cost-level beam search with prize function on 10000 cases
python random_test.py  --planlimit 20 --span 10000 --mode no --pool 58 --package-num 20 --time-limit 2 --func-type constant/uniform/distance --time-interval 0.05

# cost-level beam search with learned heuristic on 10000 cases
python random_test.py  --planlimit 20 --device cuda:0 --model [model_name] --span 10000 --package-num 20 --time-limit 2 --func-type constant/uniform/distance --time-interval 0.05
