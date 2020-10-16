#! /bin/bash

rm -rf models/multisl
python main.py ~/Desktop/MAAC/envs/mpe_scenarios/multi_speaker_listener multisl --episode_length 25 --n_rollout_threads 1 --use_gpu
