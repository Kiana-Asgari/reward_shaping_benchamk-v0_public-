from env.point_mass import ContextualPointMass
from env.delayed_env import DelayedEnv
from baseline.PPO import PPO
from utils import initialize_ppo
from modules.reward import IntrinsicReward
import torch

if __name__ == '__main__':
    args = initialize_ppo()

    sparse_env = DelayedEnv(env_id=args.env_id, use_intrinsic_reward=True, reward_freq=10, mixing_wieght=1)

    state_dim = sparse_env.observation_space.shape[0]

    action_dim = sparse_env.action_space.shape[0]

    reward_model = IntrinsicReward(action_shape=action_dim, obs_shape=state_dim, args=args)

    policy_model = PPO(state_dim=state_dim, action_dim=action_dim, args=args)

    print('===============Model Started Running================')
    for t in range(args.env_interactions):
        #########################
        # Update Policy Function#
        #########################
        print('---------------Policy Update---------------')
        # We will use the inferred reward function instead of the real one.
        sparse_env.set_intrinsic_reward(reward_model.get_intrinsic_reward_network())

        buffer, new_policy = policy_model.train(env=sparse_env, test_env=sparse_env, single_update=True, args=args)

        #########################
        #     Reward Update     #
        #########################
        print('---------------Reward Update---------------')
        reward_model.update(buffer=buffer, K_epochs=args.K_epochs, policy_model=new_policy)

        #########################
        #   Optimizer Step      #
        #########################
        policy_model.optimizer.step()
        # Copy new weights into old policy
        policy_model.policy_old.load_state_dict(policy_model.policy.state_dict())
        # clearing buffer
        policy_model.buffer.clear()

        if t % 3 == 0:
            policy_model.test(test_env=sparse_env, test_num=args.test_num)






        # print(new_policy.forward(torch.tensor([-4., 0, -4., 0])).sum())
        # print(torch.autograd.grad(new_policy.forward(torch.tensor([-4., 0, -4., 0])).sum(),
        #                          reward_model.in_reward_model.parameters()))
        # input()
