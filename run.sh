# coding:utf-8
#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

for class in 0 1 2 3 4
do
    for loss in Abs
    do
        for seed in 111 222 333
        do
            for dim in 5 10 20 50 100
            do
                python3 main.py --seed $seed --hidden_dim $dim --dataset fashion --data_type $class
            done
        done
    done
done