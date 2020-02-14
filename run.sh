#!/usr/bin/env bash
##python gen_sample_IL.py --number 1000 --threshold 0.1 --start 0 --limit 20 --name $1 --debug --device $2
##python gen_sample.py --number $1 --threshold $2 --pool-num 38 --start 0 --limit 400
#python feature_extractor_gnn.py --number $1 --threshold $2 --pool-num 36 --mode IL
#python model_gnn.py --num-epochs 60 --number $1 --threshold $2 --mode IL --batch-size 128 --device cuda:2
#python model_update.py --number $1 --threshold $2 --size 128 --device 'cuda:0' --pool-num 10 --limit 12

#python random_test.py --model IL_40000 --planlimit 20 --device cuda:2 --span 1000 --mode no --pool 58 --package-num 20 --time-limit 2 --func-type constant --time-interval 0.05
#python random_test.py --model IL_40000 --planlimit 20 --device cuda:2 --span 1000 --mode no --pool 58 --package-num 20 --time-limit 2 --func-type uniform --time-interval 0.01
#python random_test.py --model IL_40000 --planlimit 20 --device cuda:2 --span 1000 --mode no --pool 58 --package-num 20 --time-limit 2 --func-type distance --time-interval 0.01
#python random_test.py --model IL_40000 --planlimit 20 --device cuda:2 --span 1000 --mode no --pool 58 --package-num 50 --time-limit 3 --func-type constant --time-interval 0.01
#python random_test.py --model IL_40000 --planlimit 20 --device cuda:2 --span 1000 --mode no --pool 58 --package-num 50 --time-limit 3 --func-type uniform --time-interval 0.01
#python random_test.py --model IL_40000 --planlimit 20 --device cuda:2 --span 1000 --mode no --pool 58 --package-num 50 --time-limit 3 --func-type distance --time-interval 0.01
#python random_test.py --model IL_40000 --planlimit 20 --device cuda:2 --span 1000 --mode no --pool 58 --package-num 100 --time-limit 4 --func-type constant --time-interval 0.01
#python random_test.py --model IL_40000 --planlimit 20 --device cuda:2 --span 1000 --mode no --pool 58 --package-num 100 --time-limit 4 --func-type uniform --time-interval 0.01
#python random_test.py --model IL_40000 --planlimit 20 --device cuda:2 --span 1000 --mode no --pool 58 --package-num 100 --time-limit 4 --func-type distance --time-interval 0.01

#for i in 0 1 2 3 4
#do
#    python DQN-variant.py --device cuda:3 --double-dqn --lr 0.001 --package-num 50 --hidden-size 64 --func-type distance --time-limit 3  --num-env 32 --max-step 6000 --fn 64hidden$i
#done
#
#
#for i in 0 1 2 3 4
#do
#    python DQN.py --device cuda:1 --double-dqn --lr 0.001 --package-num 50 --func-type distance --time-limit 3  --num-env 32 --nlayer 3 --max-step 6000 --fn 3layer$i
#done

#for i in 0 1 2 3 4
#do
#    python DQN.py --device cuda:0 --double-dqn --lr 0.001 --package-num 50 --hidden-size 8  --func-type distance --time-limit 3  --num-env 32 --max-step 6000 --fn 8hidden$i
#done

#python DQN.py --device cpu --double-dqn --lr 0.001 --package-num 20  --func-type distance --time-limit 2  --num-env 32 --max-step 20000
#python DQN.py --device cpu --double-dqn --lr 0.001 --package-num 20  --func-type constant --time-limit 2  --num-env 32 --max-step 20000
#python DQN.py --device cpu --double-dqn --lr 0.001 --package-num 20  --func-type uniform --time-limit 2  --num-env 32 --max-step 20000
#
#python DQN.py --device cuda:0 --double-dqn --lr 0.001 --package-num 50  --func-type distance --time-limit 3  --num-env 32 --max-step 20000
#python DQN.py --device cuda:0 --double-dqn --lr 0.001 --package-num 50  --func-type constant --time-limit 3  --num-env 32 --max-step 13000
#python DQN.py --device cuda:3 --double-dqn --lr 0.001 --package-num 100  --func-type distance --time-limit 4  --num-env 32 --max-step 50000
#python DQN.py --device cuda:3 --double-dqn --lr 0.001 --package-num 100  --func-type constant --time-limit 4  --num-env 32 --max-step 50000
python DQN.py --device cuda:3 --double-dqn --lr 0.001 --package-num 100  --func-type uniform --time-limit 4  --num-env 16 --max-step 10000
python DQN.py --device cuda:3 --double-dqn --lr 0.001 --package-num 100  --func-type uniform --time-limit 4  --num-env 32 --max-step 10000 --beam