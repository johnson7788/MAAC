#! /bin/bash

rm -rf models/fullobs
python main.py ~/Desktop/MAAC/envs/mpe_scenarios/fullobs_collect_treasure fullobs --episode_length 100 --n_rollout_threads 1 --use_gpu
