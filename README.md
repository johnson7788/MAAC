# Multi-Actor-Attention-Critic

![Example](rewardfuncs/lenshipyards/example.gif)

这个资源库是我为Two Sigma的2020年Halite竞赛提交的，在这个游戏中，你必须控制多种类型的单元并做出智能决定，以便在游戏结束时积累最大数量的Halite。 
我选择了一个基于强化学习的方法，但我想通过一个分散的策略和一个集中的critic来尝试一些多agent的方法（因为Halite确实提供了完美的信息，而Battlecode等游戏则没有）。

## Please Note!

这里所做的绝大部分工作是作为[Shariq Iqbal等人的这个项目]的一部分完成的（https://github.com/shariqiqbal2810/MAAC）。
他提出了整个算法，并很好地将其背后的代码开源了。 我做了一些修改，以便使这个算法适用于更复杂的游戏，如Halite，
他们的算法在这里解释为他们论文的一部分。[*Actor-Attention-Critic for Multi-Agent Reinforcement Learning*]（https://arxiv.org/abs/1810.02912）（Iqbal and Sha，ICML 2019）。

## Modifications

- 在游戏过程中动态地添加/删除agent。
  - 在Halite中，船只可以被摧毁，转化为船坞，或从船坞中产生，所以棋盘上的船只总数是不可预测的。 
- 每个agent类型有一个网络。
  - 例如，我们可以简单地创建一个与团队无关的船舶网络，只用我们积累的所有团队的船舶处理数据来训练该网络，而不是为每个额外的 "船舶 "类型的agent创建一个新的编码器网络。

为了解决这些问题，我对训练进程进行了一定程度的重组，使策略和critic网络的总数反映了agent类型的数量，
而不是棋盘上agent的数量（q损失和策略损失简单相加）。 重放缓冲区也进行了重组，以处理每个时间段都存在未知数量的agent的事实。

测试方案在`envs/test_scenarios`中可用，以验证不同类型的agent特征。

## Citations

```
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
