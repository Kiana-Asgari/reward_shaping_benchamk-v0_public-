import argparse
import math
from random import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle



def distance(ter1, ter2, gamma):
    R1 = np.array([gamma ** i * ter1[2][i] for i in range(len(ter1[2]))]).sum()
    R2 = np.array([gamma ** i * ter2[2][i] for i in range(len(ter2[2]))]).sum()
    return R1 >= R2


def p_function(r1, r2, gamma):  # P(T1>T2)
    G1 = sum([gamma ** i * r1[i] for i in range(len(r1))])
    G2 = sum([gamma ** i * r2[i] for i in range(len(r2))])
    temp = 1 / (1 + torch.exp(G2-G1))
    if math.isnan(temp):
        print('r1', r1)
        print('r2', r2)
        print('gamma', gamma)
        print('g1=', G1)
        print('g2=', G2)
        print(temp)
        print(' p is nan!')
    return temp


def fill_trajectory_buffer(trajectory_buffer, policy_model, env, initial_trj):
    t = 0
    while t < initial_trj:  #TODO
        s, a, r, sa = policy_model.run_single_episode(env=env)
        trajectory_buffer.append([s, a, r, sa])
        t += 1
    #    t += len(s)


def initialize_ppo():

    parser = argparse.ArgumentParser(description='PyTorch PPO Args')

    parser.add_argument('--env_name', default='PointMass',
                        help='continuous Gym environment (default: Pendulum-v1)')
    parser.add_argument('--reward_freq', type=int, default=1,
                        help='frequancy of sparse reward')
    parser.add_argument('--has_continuous_action_space', type=bool, default=True)

    parser.add_argument('--env_interactions', type=int, default=250, metavar='N')

    parser.add_argument('--trj_buffer_size', type=int, default=1000, metavar='N')
    parser.add_argument('--initial_trj', type=int, default=100, metavar='N')
    parser.add_argument('--trj_batch_size', type=int, default=10, metavar='N')

    parser.add_argument('--p_update_period', type=int, default=50, metavar='N')
    parser.add_argument('--r_update_period', type=int, default=50, metavar='N')
    parser.add_argument('--p_update_num', type=int, default=10, metavar='N')
    parser.add_argument('--r_update_num', type=int, default=40, metavar='N')

    parser.add_argument('--max_ep_len', type=int, default=80, metavar='N')

    parser.add_argument('--print_freq', type=int, default=1000, metavar='N')
    parser.add_argument('--test_freq', type=int, default=4000, metavar='N')
    parser.add_argument('--test_num', type=int, default=20, metavar='N')

    parser.add_argument('--action_std', type=float, default=0.6, metavar='G')
    parser.add_argument('--action_std_init', type=float, default=0.6, metavar='G')
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05, metavar='G')
    parser.add_argument('--min_action_std', type=float, default=0.1, metavar='G')
    parser.add_argument('--action_std_decay_freq', type=int, default=int(2.5e5), metavar='N')

    parser.add_argument('--update_timestep', type=int, default=int(500*4), metavar='N')
    parser.add_argument('--max_training_timesteps', type=int, default=int(4000*30), metavar='N')

    parser.add_argument('--k_epochs', type=int, default=10, metavar='N')
    parser.add_argument('--eps_clip', type=float, default=0.2, metavar='G')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G')

    parser.add_argument('--lr_actor', type=float, default=0.0003, metavar='G')
    parser.add_argument('--lr_critic', type=float, default=0.001, metavar='G')
    parser.add_argument('--reward_lr', type=float, default=0.0003, metavar='G')

    parser.add_argument('--random_seed', type=int, default=0, metavar='N')

    args = parser.parse_args()

    return args


def initialize_sac():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

    parser.add_argument('--env-name', default="Pendulum-v1",
                        help='continuous Gym environment (default: Pendulum-v1)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--G', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='SAC learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='batch size (default: 100)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                        help='Steps sampling random actions (default: 1000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--gamma', type=int, default=0.99, metavar='N',
                        help='discounted learning rate')
    # Testing the Model
    parser.add_argument('--test_freq', type=int, default=4, metavar='N',
                        help='frequency of testing the agent (default: 10)')
    parser.add_argument('--test_num', type=int, default=4, metavar='N',
                        help='number of testing the agent (default: 10)')
    parser.add_argument('--log_dir', default='/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/SAC/',
                        help='logging directory')
    # Reward Model
    parser.add_argument('--reward_freq', type=int, default=100, metavar='N',
                        help='delaying environment reward for many time steps (default:10)')
    parser.add_argument('--env_interactions', type=int, default=int(350), metavar='N',
                        help='total number of episodes to run the main loop')
    parser.add_argument('--trj_buffer_size', type=int, default=2000, metavar='N',
                        help='replay buffer size for trajectories')
    parser.add_argument('--trj_batch_size', type=int, default=10, metavar='N',
                        help='replay buffer size for trajectories while updating reward ')

    parser.add_argument('--p_update_period', type=int, default=50, metavar='N',
                        help='period of updating the policy network')
    parser.add_argument('--r_update_period', type=int, default=50, metavar='N',
                        help='period of updating the reward network')

    parser.add_argument('--p_update_num', type=int, default=1e6, metavar='N',  # TODO
                        help='number of updating the policy network')
    parser.add_argument('--r_update_num', type=int, default=30, metavar='N',
                        help='number of updating the policy network')

    parser.add_argument('--reward_lr', type=float, default=0.0003, metavar='G',
                        help='reward learning rate (default: 0.0003)')
    parser.add_argument('--initial_trj', type=int, default=int(100000), metavar='N',
                        help='collecting initial set of trajectories')
    args = parser.parse_args()

    return args


def fix_length(a, b, c):

    n = len(c[0])
    m = len(a[0])
    print(n-m)
    for i in range(n-m):
        print(i)
        for data in range(len(a)):
            a[data].append(a[data][m-2] + random()/8)
            b[data].append(b[data][m - (n-m)+i])
    return a, b, c



def plot_learning_curve(sors_file_dir, sac_file_dir, gt_sac_file_dir, log_dir, name):
    with open(sors_file_dir, "rb") as f:
        sors_rewards = pickle.load(f)

    with open(sac_file_dir, "rb") as f:
        sac_rewards = pickle.load(f)

    with open(gt_sac_file_dir, "rb") as f:
        gt_sac_rewards = pickle.load(f)

    x = 4000 * np.array(range(len(sors_rewards[0])))
    print(len(x))

    sd_array = []
    mean_array = []

    for i in range(len(sors_rewards[0])):
        temp = []
        for data in range(len(sors_rewards)):
            temp.append(sors_rewards[data][i])
        sd_array.append(np.std(temp))
        mean_array.append(np.mean(temp))

    sors_mean = np.array(mean_array)
    sors_sd = np.array(sd_array)

    sd_array = []
    mean_array = []
    for i in range(len(sac_rewards[0])):
        temp = []
        for data in range(len(sac_rewards)):
            temp.append(sac_rewards[data][i])
        sd_array.append(np.std(temp))
        mean_array.append(np.mean(temp))

    sac_mean = np.array(mean_array)
    sac_sd = np.array(sd_array)

    sd_array = []
    mean_array = []
    for i in range(len(gt_sac_rewards[0])):
        temp = []
        for data in range(len(gt_sac_rewards)):
            temp.append(gt_sac_rewards[data][i])
        sd_array.append(np.std(temp))
        mean_array.append(np.mean(temp))

    gt_sac_mean = np.array(mean_array)
    gt_sac_sd = np.array(sd_array)

    plt.figure()

    plt.plot(x, sors_mean, label='SORS')
    plt.fill_between(x, sors_mean - sors_sd, sors_mean + sors_sd, color='b', alpha=.1)

    plt.plot(x, sac_mean, label='PPO')
    plt.fill_between(x, sac_mean - sac_sd, sac_mean + sac_sd, color='r', alpha=.1)

    plt.plot(x, gt_sac_mean, label='PPO W/ GT Reward')
    plt.fill_between(x, gt_sac_mean - gt_sac_sd, gt_sac_mean + gt_sac_sd, color='g', alpha=.1)

    plt.xlabel('timeSteps')
    plt.ylabel('accumulated return')
    plt.legend(loc='lower right')
    plt.title('SPRS on PoitMass w/ Sparse reward, context = [0., 2., 2.0]')
    plt.savefig(log_dir + name + '.png')
    #plt.show()


