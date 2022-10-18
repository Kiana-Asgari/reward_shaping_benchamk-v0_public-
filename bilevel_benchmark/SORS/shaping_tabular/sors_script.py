import numpy as np

from modules.REINFORCE import REINFORCE
from modules.REINFORCE import plot_learning_curve
from modules.replay_buffer import replay_buffer
from modules.reward import reward

from modules.reward import calculate_return
from modules.reward import distance
from modules.reward import calculate_p_fucntion
from modules.reward import calculate_trejecotry_gradiant
from modules.reward import calculate_gradiant


from env.delayed_chain import DelayedChainEnv
from env.chain import ChainEnv


def fill_trajectory_buffer(trajectory_buffer, policy_model, env, buffer_size):
    for t in range(buffer_size):
        s, a, r = policy_model.run_single_episode(env=env, reward_buffer=[], action_buffer=[], state_buffer=[])
        trajectory_buffer.append([s, a, r])


def run(policy_model, reward_model, env, test_env,
        learning_rate=0.1, gamma=0.9,
        total_episodes=2000, buffer_size=2000,
        p_update_period=10, r_update_period=10,
        p_update_num=10, r_update_num=20,
        batch_size=10):
    # Initializing the trajectory buffer
    print('==================MODEL STARTED======================')
    trajectory_buffer = replay_buffer(buffer_size)
    fill_trajectory_buffer(trajectory_buffer, policy_model, env, buffer_size)

    for t in range(total_episodes):
        # Gathering Experience
        s, a, r = policy_model.run_single_episode(env, reward_buffer=[], action_buffer=[], state_buffer=[])
        trajectory_buffer.append(value=[s, a, r])

        # Update Reward Function
        if t % r_update_period == 0:
            for _ in range(r_update_num):
                trj_batch = trajectory_buffer.sample(sample_size=batch_size)  # batch size?
                reward_model.update(trj_batch=trj_batch, learning_rate=learning_rate, gamma=gamma)
           # print('[REWARD]', (t + 1) / r_update_period, 'th reward update:\n', reward_model.reward_function)

        # Update Policy Function
        if t % p_update_period == 0:
            r = reward_model.get_reward_function()  # r in |S|*|A|
            # we will use the inferred reward function instead of the real one.
            env.set_reward(r)
            policy_model.learn(total_episodes=p_update_num, env=env,
                               learning_rate=learning_rate, log_interval=10,
                               log_num=10, gamma=gamma) # logging not implemented
            test_state, _, _ = policy_model.run_single_episode(env=test_env, reward_buffer=[], action_buffer=[], state_buffer=[])

    plot_learning_curve(policy_model.test_rewards,
                        log_dir='/Users/kasgari/work/kiana-internship_code_adishs-github/code/shaping_tabular/log/',
                        name='NEWWW chain(30)')



if __name__ == '__main__':

    n_states = 20
    n_actions = 2
    env = DelayedChainEnv(n_states=n_states, n_actions=n_actions)
    test_env = ChainEnv(n_states=n_states, n_actions=n_actions)
    policy_model = REINFORCE(env=env, test_env=test_env)

    gt_reward = env.reward

    reward_model = reward(gt_reward=gt_reward, n_states=n_states, n_actions=n_actions)

    run(env=env, test_env=test_env, policy_model=policy_model, reward_model=reward_model)







