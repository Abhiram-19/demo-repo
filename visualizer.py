import pygame
import threading
import time
import numpy as np
from environment import GridWorld
from agent import FM3QAgent

COLORS = {
    'bg': (255, 255, 255),
    'grid': (200, 200, 200),
    'team1': (0, 150, 255),
    'team2': (255, 100, 100),
    'goal': (0, 200, 0),
    'obstacle': (50, 50, 50),
    'text': (0, 0, 0),
    'panel': (240, 240, 240),
    'button_active': (0, 200, 0),
    'button_inactive': (200, 0, 0)
}


class FM3QVisualizer:
    def __init__(self, size=8, cell_size=60):
        pygame.init()
        self.size = size
        self.cell_size = cell_size
        self.screen_width = size * cell_size
        self.screen_height = size * cell_size + 60  # Extra space for control panel
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("FM3Q Multi-Agent Reinforcement Learning")

        # Initialize font
        self.font = pygame.font.Font(None, 24)

        # Initialize environment and agents
        self.env = GridWorld(size)
        self.agent1 = FM3QAgent(state_size=size * size, action_size=4, num_agents=2)
        self.agent2 = FM3QAgent(state_size=size * size, action_size=4, num_agents=2)

        # Runtime variables
        self.running = True
        self.training = False
        self.clock = pygame.time.Clock()
        self.lock = threading.Lock()

        # Training statistics
        self.episode = 0
        self.max_episodes = 100
        self.step_delay = 0.02

    def draw(self):
        """Draw the current state of the environment"""
        self.screen.fill(COLORS['bg'])
        for x in range(self.size):
            for y in range(self.size):
                rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, COLORS['grid'], rect, 1)

        for x, y in self.env.obstacles:
            rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, COLORS['obstacle'], rect)

        gx, gy = self.env.goal
        pygame.draw.circle(self.screen, COLORS['goal'],
                           (gx * self.cell_size + self.cell_size // 2,
                            gy * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)

        for i, (x, y) in enumerate(self.env.team1):
            pygame.draw.circle(self.screen, COLORS['team1'],
                               (x * self.cell_size + self.cell_size // 2,
                                y * self.cell_size + self.cell_size // 2),
                               self.cell_size // 3)

        for i, (x, y) in enumerate(self.env.team2):
            pygame.draw.circle(self.screen, COLORS['team2'],
                               (x * self.cell_size + self.cell_size // 2,
                                y * self.cell_size + self.cell_size // 2),
                               self.cell_size // 3)

        pygame.display.flip()

    def run(self):
        """Main loop handling events and drawing"""
        self.env.reset()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if self.training:
                self.train()
            self.draw()
            self.clock.tick(30)
        pygame.quit()

    def train(self):
        """Train both teams of agents using FM3Q methodology"""
        state = self.env.reset()
        done = False
        while not done:
            team1_actions = self.agent1.act(state)
            flipped_state = np.where(state == 0.5, 0.5, -state)
            team2_actions = self.agent2.act(flipped_state)

            next_state, reward, done = self.env.step(team1_actions, team2_actions)

            self.agent1.push(state, team1_actions, team2_actions, reward, next_state, done)
            self.agent2.push(flipped_state, team2_actions, team1_actions, -reward,
                             np.where(next_state == 0.5, 0.5, -next_state), done)

            self.agent1.update()
            self.agent2.update()

            state = next_state
            time.sleep(self.step_delay)