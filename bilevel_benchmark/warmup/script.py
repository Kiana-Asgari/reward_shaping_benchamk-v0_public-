from torch import optim
import torch
from torch import nn
import gym
import argparse


from env.chain import ChainEnv
from alg.tabular_REINFORCE import REINFORCE
from alg.neural_REINFORCE import n_Reinforce
from alg.neural_REINFORCE import policy_network
from alg.PPO.PPO import PPO
from alg.SAC.agent import SAC




def tabular_test():
    """testing simple chain
    environment with reinforce algorithm
    """
    n_states = 30
    n_actions = 2
    env = ChainEnv(n_states=n_states, n_actions=n_actions)
    test_env = ChainEnv(n_states=n_states, n_actions=n_actions)
    model = REINFORCE(env, test_env)
    model.learn(total_episodes=2000, log_interval=10,log_num=10, learning_rate=0.1, gamma = 0.9,
               log_dir='/Users/kasgari/work/kiana-internship_code_adishs-github/code/warmup/log/',
                name = 'Chain(30)')

def neural_test():
    env = gym.make('CartPole-v1')
    test_env = gym.make('CartPole-v1')
    lr = 0.01
    n_hidden = 16
    p = policy_network(env, n_hidden=n_hidden)
    opt = optim.Adam(p.network.parameters(), lr=lr)
    model = n_Reinforce(env, test_env, policy_network=p, optimizer=opt)

    model.learn(num_episodes=2000,
                log_dir='/Users/kasgari/work/kiana-internship_code_adishs-github/code/warmup/log/',
                name='neural_reinforce')

def PPO_test():
    env_name = 'CartPole-v1'
    has_continuous_action_space = False  # continuous action space; else discrete
    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(2e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                    has_continuous_action_space, action_std)

    ppo_agent.train(env, max_training_timesteps, max_ep_len, update_timestep, action_std_decay_freq,
          action_std_decay_rate, min_action_std, print_freq)

def SAC_test():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=int(50_001), metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--test_freq', type=int, default=5, metavar='N',
                        help='frequancy of testing the agent (default: 10)')
    parser.add_argument('--test_num', type=int, default=5, metavar='N',
                        help='number of testing the agent (default: 10)')
    parser.add_argument('--log_dir', default='/Users/kasgari/work/kiana-internship_code_adishs-github/code/warmup/log/',
                        help='logging directory')

    args = parser.parse_args()

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make("Pendulum-v1")
    test_env = gym.make("Pendulum-v1")
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Training Loop
    agent.train(env, test_env, args)
    agent.plot_learning_curve(args, name='SAC, Pendulum')

    env.close()



if __name__ == '__main__':
   # tabular_test()
   # neural_test()
   # PPO_test()
   SAC_test()



