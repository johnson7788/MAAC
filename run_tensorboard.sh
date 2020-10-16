#! /bin/bash

echo "Model?"
read model_name

tensorboard --logdir ~/Desktop/MAAC/models/$model_name
