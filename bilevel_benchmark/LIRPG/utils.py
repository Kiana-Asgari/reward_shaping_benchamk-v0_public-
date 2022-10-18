import argparse
import numpy as np
import torch
from modules.buffer import RolloutBuffer


def assign_grads(loss, network, create_graph):
    # Assigning gradients to three layered linear network
    grad = torch.autograd.grad(loss.mean(), network.parameters(), create_graph=create_graph)
    network[0].weight.grad = grad[0]
    network[0].bias.grad = grad[1]

    network[2].weight.grad = grad[2]
    network[2].bias.grad = grad[3]

    network[4].weight.grad = grad[4]
    network[4].bias.grad = grad[5]






def batch_sampling(sample_size: int, buffer: RolloutBuffer) -> RolloutBuffer:
    batch_buffer = RolloutBuffer()
    idxes = np.random.choice(len(buffer.states), sample_size)

    for idx in idxes:
        batch_buffer.actions.append(buffer.actions[idx])
        batch_buffer.states.append(buffer.states[idx])
        batch_buffer.states_actions.append(buffer.states_actions[idx])

        batch_buffer.logprobs.append(buffer.logprobs[idx])

        batch_buffer.ex_rewards.append(buffer.ex_rewards[idx])
        batch_buffer.inex_rewards.append(buffer.inex_rewards[idx])

        batch_buffer.is_terminals.append(buffer.is_terminals[idx])

    return batch_buffer


def initialize_ppo():
    parser = argparse.ArgumentParser(description='PyTorch PPO Args')

    parser.add_argument('--env_id', default='PointMass',
                        help='continuous Gym environment (default: Pendulum-v1)')
    parser.add_argument('--reward_freq', type=int, default=1,
                        help='frequency of sparse reward')
    parser.add_argument('--mix_coef', type=float, default=1,
                        help='mixing coefficient of extrinsic and intrinsic reward')

    parser.add_argument('--has_continuous_action_space', type=bool, default=True)

    parser.add_argument('--env_interactions', type=int, default=250, metavar='N')

    parser.add_argument('--trj_batch_size', type=int, default=10, metavar='N')

    parser.add_argument('--max_ep_len', type=int, default=500, metavar='N')

    parser.add_argument('--test_freq', type=int, default=2048*2, metavar='N')
    parser.add_argument('--test_num', type=int, default=20, metavar='N')

    parser.add_argument('--action_std', type=float, default=0.6, metavar='G')
    parser.add_argument('--action_std_init', type=float, default=0.6, metavar='G')
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05, metavar='G')
    parser.add_argument('--min_action_std', type=float, default=0.1, metavar='G')
    parser.add_argument('--action_std_decay_freq', type=int, default=int(2.5e5), metavar='N')

    parser.add_argument('--update_timestep', type=int, default=int(2048), metavar='N')
    parser.add_argument('--max_training_timesteps', type=int, default=int(1e7), metavar='N')

    parser.add_argument('--K_epochs', type=int, default=1, metavar='N')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N')

    parser.add_argument('--eps_clip', type=float, default=0.2, metavar='G')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G')

    parser.add_argument('--lr_actor', type=float, default=0.0003, metavar='G')
    parser.add_argument('--lr_critic', type=float, default=0.001, metavar='G')
    parser.add_argument('--lr_in_reward', type=float, default=0.0003, metavar='G')
    parser.add_argument('--lr_ex_value', type=float, default=0.001, metavar='G')

    parser.add_argument('--random_seed', type=int, default=0, metavar='N')
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()

    return args
