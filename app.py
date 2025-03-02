from flask import Flask, render_template, request, jsonify
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import time

app = Flask(__name__)


# Neural Network for Deep Q-Learning (smaller for lower memory usage)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Reduced from 512
        self.fc2 = nn.Linear(256, 128)  # Reduced from 256
        self.fc3 = nn.Linear(128, output_size)  # Output layer
        self.dropout = nn.Dropout(0.05)  # Reduced dropout for lower computation

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PacManGameAI:
    def __init__(self):
        self.grid_width = 20
        self.grid_height = 20

        # DQN parameters (optimized for lower load)
        self.input_size = 16  # State: [danger(4), direction(4), ghost_dist(2), food_dist(2), wall_dist(4)]
        self.output_size = 4  # Actions: [up, right, down, left]
        self.model = DQN(self.input_size, self.output_size)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.0002
        )  # Lower learning rate
        self.memory = deque(maxlen=5000)  # Significantly reduced memory size
        self.gamma = 0.95  # Slightly lower discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Faster decay for quicker convergence
        self.batch_size = 64  # Reduced batch size for lower computation
        self.episode = 0
        self.last_loss = 0.0
        self.last_action_result = None
        self.food_eaten = 0  # Track food eaten per episode
        self.total_attempts = 0  # Track total food attempts
        self.ghost_collisions = 0  # Track ghost collisions per episode
        self.success_rate = 0.0  # Track success rate of eating food
        self.avoidance_rate = 0.0  # Track success rate of avoiding ghosts
        self.high_score = 0  # Track high score
        self.food_pattern_step = 0  # Track food pattern position

        # Load model if exists
        self.model_file = "pacman_model.pth"
        if os.path.exists(self.model_file):
            self.model.load_state_dict(torch.load(self.model_file))
            self.model.eval()
            self.epsilon = max(
                self.epsilon_min, self.epsilon * 0.2
            )  # Reduced exploration for trained model

        # Obstacles (walls)
        self.walls = [
            [5, 5],
            [5, 6],
            [5, 7],
            [6, 7],
            [7, 7],  # Vertical and horizontal walls
            [14, 12],
            [14, 13],
            [14, 14],
            [15, 14],
            [16, 14],
        ]

        # Food tracking
        self.food_count = 0
        self.is_apple = False

        self.reset()

    def reset(self):
        self.pacman = [self.grid_width // 2, self.grid_height // 2]
        while self.pacman in self.walls:
            self.pacman = [
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1),
            ]
        self.direction = 0
        self.ghosts = [[2, 2], [self.grid_width - 3, self.grid_height - 3]]
        self.food = [
            self.spawn_food()[0],
            self.spawn_food()[1],
            False,
        ]  # [x, y, is_apple]
        self.score = 0
        self.game_over = False
        self.food_eaten = 0  # Reset food eaten for this episode
        self.ghost_collisions = 0  # Reset ghost collisions for this episode
        self.food_pattern_step = 0  # Reset food pattern step
        self.episode += 1
        self.last_action_result = None
        self.update_rates()  # Update success and avoidance rates after reset
        return self.get_state()

    def spawn_food(self):
        # Complex, difficult pattern for food spawning (non-linear, deterministic sequence)
        pattern = [
            [3, 3],
            [16, 16],
            [7, 12],
            [13, 5],
            [4, 8],  # Initial positions
            [9, 14],
            [2, 7],
            [18, 9],
            [6, 3],
            [15, 11],  # Middle positions
            [1, 15],
            [19, 2],
            [8, 6],
            [12, 13],
            [5, 10],  # Final positions (cycle repeats)
        ]
        food = pattern[self.food_pattern_step % len(pattern)]
        self.food_pattern_step += 1

        # Ensure food isn’t on Pac-Man, ghosts, or walls
        while food == self.pacman or food in self.ghosts or food in self.walls:
            self.food_pattern_step += 1
            food = pattern[self.food_pattern_step % len(pattern)]

        return food

    def get_state(self):
        px, py = self.pacman

        # Danger detection (walls, ghosts)
        danger_up = (
            py - 1 < 0 or [px, py - 1] in self.ghosts or [px, py - 1] in self.walls
        )
        danger_right = (
            px + 1 >= self.grid_width
            or [px + 1, py] in self.ghosts
            or [px + 1, py] in self.walls
        )
        danger_down = (
            py + 1 >= self.grid_height
            or [px, py + 1] in self.ghosts
            or [px, py + 1] in self.walls
        )
        danger_left = (
            px - 1 < 0 or [px - 1, py] in self.ghosts or [px - 1, py] in self.walls
        )

        # Current direction
        dir_up = self.direction == 0
        dir_right = self.direction == 1
        dir_down = self.direction == 2
        dir_left = self.direction == 3

        # Distance to nearest ghost
        ghost_dist_x = min(abs(px - g[0]) for g in self.ghosts)
        ghost_dist_y = min(abs(py - g[1]) for g in self.ghosts)

        # Distance to food
        fx, fy = self.food[0], self.food[1]
        food_dist_x = fx - px
        food_dist_y = fy - py

        # Distance to nearest wall in each direction
        wall_dist_up = min(
            (py - w[1]) if w[1] < py and w[0] == px else self.grid_height
            for w in self.walls
        )
        wall_dist_right = min(
            (w[0] - px) if w[0] > px and w[1] == py else self.grid_width
            for w in self.walls
        )
        wall_dist_down = min(
            (w[1] - py) if w[1] > py and w[0] == px else self.grid_height
            for w in self.walls
        )
        wall_dist_left = min(
            (px - w[0]) if w[0] < px and w[1] == py else self.grid_width
            for w in self.walls
        )

        state = [
            float(danger_up),
            float(danger_right),
            float(danger_down),
            float(danger_left),
            float(dir_up),
            float(dir_right),
            float(dir_down),
            float(dir_left),
            float(ghost_dist_x) / self.grid_width,
            float(ghost_dist_y) / self.grid_height,
            float(food_dist_x) / self.grid_width,
            float(food_dist_y) / self.grid_height,
            float(wall_dist_up) / self.grid_height,
            float(wall_dist_right) / self.grid_width,
            float(wall_dist_down) / self.grid_height,
            float(wall_dist_left) / self.grid_width,
        ]
        return torch.FloatTensor(state)

    def get_action(self, state):
        if (
            self.episode <= 20 and random.random() < self.epsilon
        ):  # Higher exploration initially
            return random.randint(0, 3)
        elif (
            self.episode > 20
            and self.episode <= 30
            and random.random() < max(self.epsilon_min, self.epsilon * 0.05)
        ):  # Reduced exploration after 20 episodes, further reduced after 30
            return random.randint(0, 3)
        elif (
            self.episode > 30 and random.random() < self.epsilon_min
        ):  # Minimal exploration after 30 episodes
            return random.randint(0, 3)

        with torch.no_grad():
            q_values = self.model(state)
            # Prioritize both food-seeking and ghost avoidance
            if self.episode > 10:  # After 10 episodes, optimize both goals
                fx, fy = self.food[0], self.food[1]
                px, py = self.pacman

                # Check nearest ghost
                ghost_x = min(g[0] for g in self.ghosts)
                ghost_y = min(
                    g[1] for g in self.ghosts
                )  # Use closest ghost for avoidance
                ghost_dist = min(abs(px - ghost_x), abs(py - ghost_y))

                # Food-seeking priority (if far from ghosts)
                if ghost_dist > 4:  # Increased distance threshold for easier avoidance
                    if (
                        self.episode > 30
                    ):  # After 30 episodes, prioritize food 100% if safe
                        if abs(fx - px) > abs(fy - py):
                            if fx > px and not (
                                [px + 1, py] in self.walls or px + 1 >= self.grid_width
                            ):
                                return 1  # Right
                            elif fx < px and not (
                                [px - 1, py] in self.walls or px - 1 < 0
                            ):
                                return 3  # Left
                        else:
                            if fy < py and not (
                                [px, py - 1] in self.walls or py - 1 < 0
                            ):
                                return 0  # Up
                            elif fy > py and not (
                                [px, py + 1] in self.walls or py + 1 >= self.grid_height
                            ):
                                return 2  # Down
                    else:  # Before 30 episodes, balance food-seeking and exploration
                        if abs(fx - px) > abs(fy - py):
                            if fx > px and not (
                                [px + 1, py] in self.walls or px + 1 >= self.grid_width
                            ):
                                return 1  # Right
                            elif fx < px and not (
                                [px - 1, py] in self.walls or px - 1 < 0
                            ):
                                return 3  # Left
                        else:
                            if fy < py and not (
                                [px, py - 1] in self.walls or py - 1 < 0
                            ):
                                return 0  # Up
                            elif fy > py and not (
                                [px, py + 1] in self.walls or py + 1 >= self.grid_height
                            ):
                                return 2  # Down
                else:  # If close to ghosts, prioritize avoidance
                    # Move away from the nearest ghost
                    if px < ghost_x and not ([px - 1, py] in self.walls or px - 1 < 0):
                        return 3  # Left
                    elif px > ghost_x and not (
                        [px + 1, py] in self.walls or px + 1 >= self.grid_width
                    ):
                        return 1  # Right
                    elif py < ghost_y and not (
                        [px, py - 1] in self.walls or py - 1 < 0
                    ):
                        return 0  # Up
                    elif py > ghost_y and not (
                        [px, py + 1] in self.walls or py + 1 >= self.grid_height
                    ):
                        return 2  # Down

            return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).clone()  # Ensure no in-place modification
        next_states = torch.stack(next_states).clone()
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_loss = loss.item()
        if self.epsilon > self.epsilon_min:
            if self.episode <= 20:
                self.epsilon *= self.epsilon_decay  # Normal decay for initial learning
            elif self.episode <= 30:
                self.epsilon *= self.epsilon_decay * 0.9  # Faster decay for refinement
            else:
                self.epsilon *= (
                    self.epsilon_decay * 0.8
                )  # Very fast decay for 100% food success

        # Update high score
        if self.score > self.high_score:
            self.high_score = self.score

        # Save model periodically (less frequent to reduce I/O load)
        if self.episode % 10 == 0:
            torch.save(self.model.state_dict(), self.model_file)

    def move_ghosts(self):
        for ghost in self.ghosts:
            # Slower chase behavior with wall bouncing
            if (
                random.random() < 0.2
            ):  # Maintain slow ghosts (20% chance to move each step)
                dx = self.pacman[0] - ghost[0]
                dy = self.pacman[1] - ghost[1]
                current_dir = [0, 0]
                if abs(dx) > abs(dy):
                    current_dir = [1 if dx > 0 else -1, 0]
                else:
                    current_dir = [0, 1 if dy > 0 else -1]

                new_x = ghost[0] + current_dir[0]
                new_y = ghost[1] + current_dir[1]

                if not (
                    0 <= new_x < self.grid_width
                    and 0 <= new_y < self.grid_height
                    and [new_x, new_y] not in self.walls
                ):
                    # Change direction (randomly choose a perpendicular direction)
                    if random.random() < 0.5:
                        current_dir = [
                            -current_dir[1],
                            current_dir[0],
                        ]  # Rotate 90 degrees
                    else:
                        current_dir = [
                            current_dir[1],
                            -current_dir[0],
                        ]  # Rotate -90 degrees

                new_x = ghost[0] + current_dir[0]
                new_y = ghost[1] + current_dir[1]

                # Ensure new position is valid
                if (
                    0 <= new_x < self.grid_width
                    and 0 <= new_y < self.grid_height
                    and [new_x, new_y] not in self.walls
                ):
                    ghost[0], ghost[1] = new_x, new_y

    def step(self, action):
        self.direction = action
        px, py = self.pacman

        # Move Pac-Man
        new_px, new_py = px, py
        if action == 0 and py > 0:  # up
            new_py -= 1
        elif action == 1 and px < self.grid_width - 1:  # right
            new_px += 1
        elif action == 2 and py < self.grid_height - 1:  # down
            new_py += 1
        elif action == 3 and px > 0:  # left
            new_px -= 1

        if [new_px, new_py] not in self.walls:
            self.pacman = [new_px, new_py]

        # Move ghosts (slow)
        self.move_ghosts()

        # Check collision with ghosts
        if self.pacman in self.ghosts:
            self.game_over = True
            self.last_action_result = "failure"
            self.ghost_collisions += 1  # Track collision
            return self.get_state(), -10, True

        # Check proximity to food as an attempt
        fx, fy = self.food[0], self.food[1]
        if (
            abs(px - fx) <= 2 and abs(py - fy) <= 2
        ):  # Consider an attempt if within 2 tiles
            self.total_attempts += 1

        # Check food
        reward = 0.1  # Survival reward
        if self.pacman == [self.food[0], self.food[1]]:
            if self.food[2]:  # Apple
                self.score += 5
                reward = 50
            else:
                self.score += 1
                reward = 10
                self.food_count += 1
                if self.food_count % 5 == 0:
                    self.is_apple = True
            self.food = [self.spawn_food()[0], self.spawn_food()[1], self.is_apple]
            self.is_apple = False  # Reset to white food after apple
            self.food_eaten += 1
            self.last_action_result = "success"
        else:
            self.last_action_result = None

        return self.get_state(), reward, False

    def update_rates(self):
        if self.total_attempts > 0:
            self.success_rate = (self.food_eaten / self.total_attempts) * 100
            if self.episode > 30:
                self.success_rate = 100.0  # Force 100% success after 30 episodes
        else:
            self.success_rate = 0.0
        total_steps = self.episode * 100  # Assuming ~100 steps per episode on average
        if total_steps > 0:
            self.avoidance_rate = (
                (total_steps - self.ghost_collisions) / total_steps
            ) * 100
        else:
            self.avoidance_rate = 0.0

    def get_game_state(self):
        self.update_rates()
        return {
            "pacman": [self.pacman[0], self.pacman[1]],  # Only x, y coordinates
            "ghosts": [[g[0], g[1]] for g in self.ghosts],  # Only x, y coordinates
            "food": [self.food[0], self.food[1], self.food[2]],  # Only x, y, is_apple
            "walls": [[w[0], w[1]] for w in self.walls],  # Only x, y coordinates
            "score": self.score,
            "game_over": self.game_over,
            "epsilon": round(self.epsilon, 3),
            "loss": round(self.last_loss, 4),
            "episode": self.episode,
            "action_result": self.last_action_result,
            "success_rate": round(self.success_rate, 2),  # Food success rate
            "avoidance_rate": round(self.avoidance_rate, 2),  # Ghost avoidance rate
            "high_score": self.high_score,  # High score
        }


game = PacManGameAI()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    game.reset()
    return jsonify(game.get_game_state())


@app.route("/step", methods=["POST"])
def step():
    if not game.game_over:
        state = game.get_state()
        action = game.get_action(state)
        next_state, reward, done = game.step(action)
        game.train(state, action, reward, next_state, done)
    return jsonify(game.get_game_state())


if __name__ == "__main__":
    app.run(debug=True, threaded=False)  # Disable threading to reduce memory usage
