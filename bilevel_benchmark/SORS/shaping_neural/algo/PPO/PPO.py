import torch
import torch.nn as nn
import numpy as np
from algo.PPO.buffer import RolloutBuffer
from algo.PPO.network import ActorCritic

class PPO:
    def __init__(self, state_dim, action_dim, test_env, args, device=torch.device('cpu')):

        self.test_env = test_env

        self.has_continuous_action_space = args.has_continuous_action_space

        if self.has_continuous_action_space:
            self.action_std = args.action_std_init

        self.gamma = args.gamma
        self.eps_clip = args.eps_clip

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, args.has_continuous_action_space, args.action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': args.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': args.lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, args.has_continuous_action_space, args.action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.device = device

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
            if (self.action_std <= min_action_std):
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

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self, K_epochs):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(K_epochs):
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

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def train(self, env, test_env, args):

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        # training loop
        while time_step <= args.max_training_timesteps:

            state = env.reset()
            current_ep_reward = 0

            for t in range(1, args.max_ep_len + 1):  # Run single episode

                # select action with policy
                action = self.select_action(state)
                state, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % args.update_timestep == 0:
                    self.update(args.k_epochs)

                if time_step % args.test_freq == 0:
                    self.test(self.test_env, args.test_num)

                # if continuous action space; then decay action std of ouput action distribution
                if self.has_continuous_action_space and time_step % args.action_std_decay_freq == 0:
                    self.decay_action_std(args.action_std_decay_rate, args.min_action_std)

                    print_running_reward = 0
                    print_running_episodes = 0

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

    def test(self, test_env, test_num):
        _return = 0

        for i in range(test_num):
           s, a, r, sa = self.run_single_episode(test_env)
           _return += sum([r[t]*self.gamma**t for t in range(len(r))])
        print('        in test', _return / test_num)
        self.test_rewards.append(_return / test_num)


    def run_single_episode(self, env):

        s_0 = env.reset()
        s = []
        a = []
        r = []
        sa = []
        _done = False

        while not _done:
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

            s_1, _reward, _done, _ = env.step(action)

            s.append(s_0)
            r.append(_reward)
            a.append(action)
            sa.append(np.append(state, action))
            s_0 = s_1

        return s, a, r, sa



