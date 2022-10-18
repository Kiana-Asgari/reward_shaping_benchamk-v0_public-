import torch
import torch.nn as nn
import numpy as np

from modules.buffer import RolloutBuffer
from modules.policies import ActorCritic, NewPolicy
from utils import batch_sampling, assign_grads


class PPO:
    def __init__(self, state_dim, action_dim, args):

        self.has_continuous_action_space = args.has_continuous_action_space
        self.action_std_init = args.action_std_init

        if self.has_continuous_action_space:
            self.action_std = args.action_std_init

        self.args = args

        self.gamma = args.gamma
        self.eps_clip = args.eps_clip

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, args.has_continuous_action_space, args.action_std_init).to(
            args.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': args.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': args.lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, args.has_continuous_action_space, args.action_std_init).to(
            args.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.device = args.device

        self.test_rewards = []

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.states_actions.append(torch.FloatTensor(np.append(state, action)))

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def flow_gradiant(self):
        print(self.optimizer.state_dict)
        input()

        l1_weights = self.policy.actor[0].weight.detach() - self.args.lr_actor * self.policy.actor[0].weight.grad
        l2_weights = self.policy.actor[2].weight.detach() - self.args.lr_actor * self.policy.actor[2].weight.grad
        l3_weights = self.policy.actor[4].weight.detach() - self.args.lr_actor * self.policy.actor[4].weight.grad

        l1_bias = self.policy.actor[0].bias.detach() - self.args.lr_actor * self.policy.actor[0].bias.grad
        l2_bias = self.policy.actor[2].bias.detach() - self.args.lr_actor * self.policy.actor[2].bias.grad
        l3_bias = self.policy.actor[4].bias.detach() - self.args.lr_actor * self.policy.actor[4].bias.grad

        new_policy = NewPolicy(state_dim=self.state_dim, action_dim=self.action_dim, action_std_init=self.action_std_init,
                               layer1_weights=l1_weights,  # Linear 1
                               layer2_weights=l2_weights,  # Linear 2
                               layer3_weights=l3_weights,  # Linear 3
                               layer1_bias=l1_bias,
                               layer2_bias=l2_bias,
                               layer3_bias=l3_bias)

        return new_policy

    def update(self, single_update, K_epochs, batch_size, env):
        # batch_buffer = batch_sampling(buffer=self.buffer, sample_size=batch_size)
        rewards = []
        discounted_reward = 0

        # GAE
        for reward, is_terminal in zip(reversed(self.buffer.inex_rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        if single_update:
            rewards = torch.stack(rewards).to(self.device)
        else:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        #for _ in range(K_epochs):
        # Evaluating old actions and values
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values)

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - old_logprobs.detach())

        # Finding Surrogate Loss
        advantages = rewards - state_values.detach()

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        # final loss of clipped objective PPO
        loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

        # take gradient step # TODO speed
        self.optimizer.zero_grad(set_to_none=True)
        assign_grads(loss.mean(), self.policy.actor, create_graph=True)
        assign_grads(loss.mean(), self.policy.critic, create_graph=True)

        new_policy = self.flow_gradiant()

        if not single_update:
            self.optimizer.step()
            # clearing buffer
            self.buffer.clear()
            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())


        return self.buffer, new_policy
        # return self.buffer, None

    def train(self, env, test_env, single_update, args):

        time_step = 0

        # training loop
        while time_step <= args.max_training_timesteps:

            state = env.reset()
            current_ep_reward = 0

            for t in range(1, args.max_ep_len + 1):  # Run single episode

                # select action with policy
                action = self.select_action(state)
                state, inex_reward, done, info = env.step(action)

                # saving reward and is_terminals
                self.buffer.inex_rewards.append(inex_reward)
                self.buffer.ex_rewards.append(info['ex_reward'])
                self.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += inex_reward

                # update PPO agent
                if time_step % args.update_timestep == 0:
                    temp_buffer, new_policy = self.update(single_update=single_update, K_epochs=args.K_epochs,
                                                          batch_size=args.batch_size,
                                                          env=env)

                    if single_update:
                        return temp_buffer, new_policy


                # test PPO agent
                if time_step % args.test_freq == 0:
                    self.test(test_env, args.test_num)

                # if continuous action space; then decay action std of output action distribution
                if self.has_continuous_action_space and time_step % args.action_std_decay_freq == 0:
                    self.decay_action_std(args.action_std_decay_rate, args.min_action_std)

                # break; if the episode is over
                if done:
                    break


    def test(self, test_env, test_num):
        inex_return = 0
        in_return = 0
        ex_return = 0
        for i in range(test_num):
            s, a, r_inex, r_in, r_ex, sa = self.run_single_episode(test_env)
            inex_return += sum(r_inex)
            in_return += sum(r_in)
            ex_return += sum(r_ex)

        print('---------------test-----------------')
        print('        inex:', inex_return / test_num)
        print('          ex:', ex_return / test_num)
        print('          in:', in_return / test_num)

        self.test_rewards.append(ex_return / test_num)

    def run_single_episode(self, env):
        s_0 = env.reset()
        s = []
        a = []
        r_inex = []
        r_in = []
        r_ex = []
        sa = []

        done = False
        while not done:
            if self.has_continuous_action_space:
                with torch.no_grad():
                    state = torch.FloatTensor(s_0).to(self.device)
                    action, _ = self.policy_old.act(state)
                action = action.detach().cpu().numpy().flatten()

            else:
                with torch.no_grad():
                    state = torch.FloatTensor(s_0).to(self.device)
                    action, _ = self.policy_old.act(state)
                action = action.item()

            s_1, reward, done, info = env.step(action)

            r_inex.append(reward)
            r_in.append(info['in_reward'])
            r_ex.append(info['ex_reward'])

            s.append(s_0)
            a.append(action)
            sa.append(np.append(state, action))

            s_0 = s_1
        return s, a, r_inex, r_in, r_ex, sa
