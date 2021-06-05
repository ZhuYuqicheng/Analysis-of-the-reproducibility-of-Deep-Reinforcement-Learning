# Research internship - Requirement Analysis Evaluation of Deep Reinforcement Learning Methods

## Introduction

In order to solve complex control problem in high dimensional spaces, Deep Reinforcement Learning (DRL) has been developed in recent years. However, the **reproducibility** of DRL algorithms is tricky. There are several factors that can have impact on the overall perfoemance significantly, even for exactly the same underlying algorithms [1]. In addition, most of the papers in DRL area focus on explaining the new structure of the algorithms or the novel part rather than giving the whole detail of the code that was used to generate the results. Some implementation details could cause visible deviation for the same algorithm.

In this reasearch intership, more implementation details that could influence the overall performance were investigated. Activation function for the network and the optimizer were considered as influence factors. Proximal Policy Optimization (PPO) - as the most popular Deep Reinforcement Algorithm for real world application was used in this experiment. The overall performance (reward) was tested on two different GYM environment: "CartPole-V1" and "HalfCheetah-v2".

## Environment Information

### CartPole - V1

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the **goal is to prevent it from falling over**. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

![CartPoleVideo](./record/PPO_cart_results/PPO_CartPole-v1-step-0-to-step-350.mov)

## Algorithm Information

## Experiments Results

![ActivationCart](./pictures/ActivationCartPole.png)

![ReLUCart](./pictures/ReLUCartPole.png)

![LeakyReLUCart](./pictures/LeakyReLUCartPole.png)

![ActivationCheetah](./pictures/ActivationCheetah.png)

![OptimizerCart](./pictures/OptimizerCartPole.png)

## Conclusion

## Literature

[1] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger "*Deep Reinforcement Learning that Matters*", Thirthy-Second AAAI Conference On Artificial Intelligence (AAAI), 2018, arXiv:1709.06560
