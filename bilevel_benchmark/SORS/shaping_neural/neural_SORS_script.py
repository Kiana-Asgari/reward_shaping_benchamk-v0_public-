import numpy as np
import math
import pickle
import torch
from algo.PPO.PPO import PPO
from algo.SAC.agent import SAC
from modules.reward import DiscreteReward, ContinuousReward
from modules.replay_buffer import replay_buffer
from env.delayedenv import DelayedEnv
from env.point_mass import ContextualPointMass


from utils import plot_learning_curve, initialize_sac, initialize_ppo, fill_trajectory_buffer





def run(policy_model, reward_model, env, sparse_env, args):
    print('===============Model Started Running================')

    # Initializing the trajectory buffer
    trajectory_buffer = replay_buffer(args.trj_buffer_size)
    fill_trajectory_buffer(trajectory_buffer, policy_model, sparse_env, args.initial_trj)  # wrt. sparse gt reward
    print('-------Trajectory Replay Buffer Initialized---------')

    for t in range(args.env_interactions):
        # Gathering New Experience
        states, actions, rewards, state_actions = policy_model.run_single_episode(sparse_env)  # wrt. sparse gt reward
        trajectory_buffer.append(value=[states, actions, rewards, state_actions])

        # Update Reward Function
        if t % args.r_update_period == 0:
            print('--------------Reward Update----------------')
            for sh in range(args.r_update_num):
                trj_batch = trajectory_buffer.sample(sample_size=args.trj_batch_size)

                reward_model.update(trj_batch=trj_batch)
                reward_model.test(trj_batch)
                print(sh, torch.tensor(reward_model.reward_network.network[0].weight)[0])

                if math.isnan(torch.tensor(reward_model.reward_network.network[0].weight)[0][0]):
                    print('nan')
                    with open(
                            "/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/TEST.pkl",
                            "rb") as f:
                        b = pickle.load(f)
                    b.append(trj_batch)

                    with open(
                            "/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/TEST.pkl",
                            "wb") as f:
                        pickle.dump(b, f)
                    input()

                #######
                """"
                with open(
                        "/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/TEST.pkl",
                        "rb") as f:
                    b = pickle.load(f)
                b.append(trj_batch)

                with open(
                        "/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/TEST.pkl",
                        "wb") as f:
                    pickle.dump(b, f)
                """
                #########



        # Update Policy Function
        if t % args.p_update_period == 0:
            print('---------------Policy Update---------------')
            # we will use the inferred reward function instead of the real one.
            env.set_reward(reward_network=reward_model.get_reward_network())
            policy_model.train(env=env, test_env=sparse_env, args=args)
            #policy_model.test(test_env=sparse_env, test_num=args.test_num)


def test_PPO_SORS():
    args = initialize_ppo()
    sparse_env = DelayedEnv(env_id=args.env_name,
                            use_dynamic_reward_function=False,
                            is_continuous=True,
                            reward_freq=args.reward_freq)

    state_dim = sparse_env.observation_space.shape[0]

    action_dim = sparse_env.action_space.shape[0]

    policy_model = PPO(state_dim, action_dim, test_env=sparse_env, args=args)

    reward_model = ContinuousReward(action_shape=action_dim, obs_shape=state_dim, args=args)

    env = DelayedEnv(env_id=args.env_name,
                     reward_network=reward_model.get_reward_network(),
                     use_dynamic_reward_function=True,
                     is_continuous=True,
                     reward_freq=args.reward_freq)



    #TEST
    policy_model.train(env=sparse_env, test_env=sparse_env, args=args)

    #run(policy_model, reward_model, env=env, sparse_env=sparse_env, args=args)

    return policy_model.test_rewards



def test_SAC_SORS():
    args = initialize_sac()
    # Ground Truth Reward Function
    sparse_env = DelayedEnv(env_id=args.env_name,
                            use_dynamic_reward_function=False,
                            is_continuous=True,
                            reward_freq=args.reward_freq)

    state_dim = sparse_env.observation_space.shape[0]
    action_dim = sparse_env.action_space.shape[0]

    # Agent
    policy_model = SAC(sparse_env.observation_space.shape[0], sparse_env.action_space, args)

    # Reward Model
    reward_model = ContinuousReward(obs_shape=state_dim, action_shape=action_dim, args=args)

    # Delayed Environment
    env = DelayedEnv(env_id=args.env_name,
                     reward_network=reward_model.get_reward_network(),
                     use_dynamic_reward_function=True,
                     is_continuous=True,
                     reward_freq=args.reward_freq)

   # run(policy_model, reward_model, env=env, sparse_env=sparse_env, args=args)



    # SAC test
    print('--test started--')
    test_env = ContextualPointMass()
    print('--env intialized--')
    policy_model = SAC(test_env.observation_space.shape[0], test_env.action_space, args)
    print('--policy model initialized--')
    policy_model.train(env=test_env, test_env=test_env, args=args)
    return policy_model.test_rewards

    #plot_learning_curve(policy_model.test_rewards,
    #                   log_dir='/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/',
    #                    name='WHATREVERSORS+SAC, pendulum, rew_freq=100 ')


if __name__ == '__main__':
    #with open("/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/"
    #          "GT_PPO+POINTMASS.pkl",
    #      "wb") as f:
    #    pickle.dump([], f)
    #    input()

   # test_PPO_SORS()


    """
    for i in range(1):
        print('=============start============')

        test_rewards = test_PPO_SORS()

        with open("/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/GT_PPO+POINTMASS.pkl",
                "rb") as f:
            pre_rewards = pickle.load(f)
        pre_rewards.append(test_rewards)

        with open("/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/GT_PPO+POINTMASS.pkl",
                "wb") as f:
                pickle.dump(pre_rewards, f)
        print(pre_rewards)
    """



    # Ploting the Results
    plot_learning_curve(sors_file_dir="/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/SORS+PPO+POINTMASS.pkl",
                        sac_file_dir="/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/PPO+POINTMASS.pkl",
                        gt_sac_file_dir="/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/GT_PPO+POINTMASS.pkl",
                        log_dir="/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_neural/log/point_mass/",
                        name='SORS+pointmass')




