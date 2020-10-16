# Multi-Actor-Attention-Critic

![Example](rewardfuncs/lenshipyards/example.gif)

This repository was intended to be my submission for Two Sigma's 2020 Halite competition, a game in which you must control multiple unit types and make intelligent decisions in order to accumulate the maximum amount of halite by the end of the game.  I elected to try a reinforcement learning based approach, but wanted to try something which was multi-agent via a decentralized policy and a centralized critic (since Halite does provide perfect information, whereas games such as Battlecode do not).  

## Please Note!

The vast majority of the work done here was completed as part of [this project by Shariq Iqbal et al.](https://github.com/shariqiqbal2810/MAAC), who came up with the entire algorithm, and was kind enough to open-source the code behind it.  I've made a couple modifications in order to make this algorithm work with a more complex game such as Halite, but the power of their algorithm work is explained here as part of their paper: [*Actor-Attention-Critic for Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1810.02912) (Iqbal and Sha, ICML 2019).  

## Modifications

- Adding/removing agents dynamically over the course of a game:
  - In Halite, ships can be destroyed, convert into shipyards, or be spawned from shipyards, so the total number of ships on the board is unpredictable.  
- One network per agent type:
  - Rather than creating a new encoder network for each additional agent of type "ship", for example, we can simply create a ship network which is team-agnostic, and train only that network with all of the data we accumulate for ship handling from all teams.  

To resolve these issues, I've restructured the training process somewhat so that the total number of policy and critic networks reflects the number of agent types, rather than the number of agents on the board (q losses and policy losses are simply summed).  The replay buffer has also been restructured to handle the fact that an unknown number of agents will exist at every timeframe.  

Test scenarios are available in `envs/test_scenarios` to validate different types of agent functionality.  

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
