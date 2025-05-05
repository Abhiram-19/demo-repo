import numpy as np


class GridWorld:
    def __init__(self, size=8):
        self.size = size
        self.action_size = 4
        self.obstacles = [(3, 3), (3, 4), (4, 3), (4, 4)]
        self.reset()

    def reset(self):
        corners = [[1, 1], [1, self.size - 2], [self.size - 2, 1], [self.size - 2, self.size - 2]]
        np.random.shuffle(corners)
        self.team1 = [corners[0], corners[1]]
        self.team2 = [corners[2], corners[3]]
        self.goal = (self.size // 2, self.size // 2)
        self.steps = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.size, self.size))
        for pos in self.team1:
            x, y = pos
            state[x, y] = 1
        for pos in self.team2:
            x, y = pos
            state[x, y] = -1
        gx, gy = self.goal
        state[gx, gy] = 0.5
        return state.flatten()

    def step(self, team1_actions, team2_actions):
        self.steps += 1
        self._move_agents(self.team1, team1_actions)
        self._move_agents(self.team2, team2_actions)

        reward = 0
        done = False
        for agent in self.team1:
            if tuple(agent) == self.goal:
                reward += 10
                done = True
        for agent in self.team2:
            if tuple(agent) == self.goal:
                reward -= 10
                done = True

        done = done or self.steps >= 100
        return self.get_state(), reward, done

    def _move_agents(self, agents, actions):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i, agent in enumerate(agents):
            if 0 <= i < len(actions):
                dx, dy = directions[actions[i]]
                new_x, new_y = agent[0] + dx, agent[1] + dy
                if 0 <= new_x < self.size and 0 <= new_y < self.size and (new_x, new_y) not in self.obstacles:
                    agent[0], agent[1] = new_x, new_y