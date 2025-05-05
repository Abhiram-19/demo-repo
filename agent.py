import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, state, action_onehot):
        x = torch.cat((state, action_onehot), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_hidden_dim=32):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)

        self.hyper_w2 = nn.Linear(state_dim, mixing_hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

        self.elu = nn.ELU()

    def forward(self, agent_qs, state):
        batch_size = state.size(0)

        w1 = torch.abs(self.hyper_w1(state))
        b1 = self.hyper_b1(state)

        w1 = w1.view(batch_size, self.n_agents, -1)
        agent_qs = agent_qs.unsqueeze(1)

        hidden = torch.bmm(agent_qs, w1).squeeze(1) + b1
        hidden = self.elu(hidden)

        w2 = torch.abs(self.hyper_w2(state)).unsqueeze(-1)
        b2 = self.hyper_b2(state)

        y = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        return y.squeeze(-1)


class FM3QAgent:
    def __init__(self, state_size, action_size, num_agents, lr=0.001, gamma=0.99, tau=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.tau = tau

        self.device = torch.device("cpu")

        self.q_nets = [QNetwork(state_size, action_size).to(self.device) for _ in range(num_agents)]
        self.target_nets = [QNetwork(state_size, action_size).to(self.device) for _ in range(num_agents)]
        for target, net in zip(self.target_nets, self.q_nets):
            target.load_state_dict(net.state_dict())

        self.optimizers = [optim.Adam(net.parameters(), lr=lr) for net in self.q_nets]
        self.replay_buffer = ReplayBuffer()

        self.mixer = QMixer(num_agents, state_size).to(self.device)
        self.target_mixer = QMixer(num_agents, state_size).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=lr)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actions = []
        for i in range(self.num_agents):
            if random.random() < self.epsilon:
                actions.append(random.randint(0, self.action_size - 1))
            else:
                q_vals = []
                for a in range(self.action_size):
                    a_onehot = torch.zeros(1, self.action_size).to(self.device)
                    a_onehot[0, a] = 1
                    q_val = self.q_nets[i](state, a_onehot)
                    q_vals.append(q_val)
                best_action = torch.argmax(torch.tensor(q_vals).to(self.device)).item()
                actions.append(best_action)
        return actions

    def _one_hot(self, actions):
        one_hots = torch.zeros(len(actions), self.num_agents, self.action_size).to(self.device)
        for i in range(len(actions)):
            for j in range(self.num_agents):
                one_hots[i, j, actions[i][j]] = 1
        return one_hots

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        samples = self.replay_buffer.sample(batch_size)
        states, t1_actions, t2_actions, rewards, next_states, dones = zip(*samples)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        t1_acts_oh = self._one_hot(t1_actions)
        agent_qs = [self.q_nets[i](states, t1_acts_oh[:, i, :]) for i in range(self.num_agents)]
        agent_qs = torch.stack(agent_qs, dim=1)

        joint_q = self.mixer(agent_qs, states)

        with torch.no_grad():
            next_agent_qs = [torch.max(self.target_nets[i](next_states, t1_acts_oh[:, i, :]), dim=1)[0] for i in
                             range(self.num_agents)]
            next_agent_qs = torch.stack(next_agent_qs, dim=1)
            target_joint_q = self.target_mixer(next_agent_qs, next_states)
            targets = rewards + self.gamma * target_joint_q * (1 - dones)

        loss = nn.MSELoss()(joint_q, targets)
        self.mixer_optimizer.zero_grad()
        loss.backward()
        self.mixer_optimizer.step()

        for net, target in zip(self.q_nets, self.target_nets):
            for target_param, param in zip(target.parameters(), net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def push(self, *args):
        self.replay_buffer.push(*args)