from env.point_mass import ContextualPointMass
from env.delayed_env import DelayedEnv
from baseline.PPO import PPO
from utils import initialize_ppo

import gym


if __name__ == '__main__':

    args = initialize_ppo()

    env = DelayedEnv(env_id=args.env_id, use_intrinsic_reward=False, reward_freq=200, mixing_wieght=args.mix_coef)
    #env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy_model = PPO(state_dim, action_dim, args=args)
    policy_model.train(env=env, test_env=env, single_update=False, args=args)

