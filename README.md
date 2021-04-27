# Multi-Actor-Attention-Critic  多Agent注意力机制的Actor-Critic
Code for [*Actor-Attention-Critic for Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1810.02912) (Iqbal and Sha, ICML 2019)

## Requirements
这些版本只是我所使用的，不一定是严格的要求。
* Python 3.6.1 (Minimum)
* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 0.3.0.post4
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and 
* [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging)

https://github.com/johnson7788/baselines
https://github.com/johnson7788/multiagent-particle-envs

## How to Run

所有训练代码都包含在`main.py`中。要查看选项，只需运行。
```shell
python main.py --help
```

# 训练模型
python main.py --env_id fullobs_collect_treasure --model_name output

我们论文中的 "合作宝藏收集游戏"环境在此版本中被称为`fullobs_collect_treasure`，而 "Rover-Tower "被称为`multi_speaker_listener`。
为了配合我们的实验，最大的episode长度应该设置为：《合作藏宝》100集，《Rover-Tower》25集。

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```bibtex
@InProceedings{pmlr-v97-iqbal19a,
  title =    {Actor-Attention-Critic for Multi-Agent Reinforcement Learning},
  author =   {Iqbal, Shariq and Sha, Fei},
  booktitle =    {Proceedings of the 36th International Conference on Machine Learning},
  pages =    {2961--2970},
  year =     {2019},
  editor =   {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume =   {97},
  series =   {Proceedings of Machine Learning Research},
  address =      {Long Beach, California, USA},
  month =    {09--15 Jun},
  publisher =    {PMLR},
  pdf =      {http://proceedings.mlr.press/v97/iqbal19a/iqbal19a.pdf},
  url =      {http://proceedings.mlr.press/v97/iqbal19a.html},
}
```
