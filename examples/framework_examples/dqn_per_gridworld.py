from machin.frame.algorithms import DQNPer
from machin.utils.logging import default_logger as logger
from machin.frame.buffers.prioritized_buffer import PrioritizedBuffer
import torch as t
import torch.nn as nn
import gym
import numpy as np
import random
from gym_minigrid.wrappers import NESWActionsImage

# configurations
env = gym.make("MiniGrid-NineRoomsDet-v4")
env = NESWActionsImage(env, max_num_actions=50)
observe_dim = [19, 19, 3]
action_num = 4
max_episodes = 200
max_steps = 50
solved_reward = 200
solved_repeat = 5

# Fixing all seed for reproducibility
seed = 1
np.random.seed(seed)
t.manual_seed(seed)
random.seed(seed)
env.seed(seed)

# model definition
class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_num)

    def forward(self, state):
        a = t.selu(self.fc1(state))
        a = t.selu(self.fc2(a))
        return self.fc3(a)

class image_Qnet(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=1, stride=1, padding=0, dilation=1):
            return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[0], kernel_size=1), kernel_size=3, padding=0),
                                kernel_size=3, padding=0)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_dim[1], kernel_size=1), kernel_size=3, padding=0),
                                kernel_size=3, padding=0)
        linear_input_size = convw * convh * 16
        print(linear_input_size)
        self.head = nn.Linear(linear_input_size, 16)
        self.out = nn.Linear(16, action_num)

    def forward(self, state):
        obs = state.permute(0, 3, 1, 2)
        x = t.selu(self.conv1(obs))
        x = t.selu(self.conv2(x))
        x = t.selu(self.conv3(x))
        x = t.selu(self.head(x.view(x.size(0), -1)))
        return self.out(x)

class deepmind_Qnet(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 19, stride=1, padding=1)

        def conv2d_size_out(size, kernel_size=1, stride=1, padding=0, dilation=1):
            return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

        convw = conv2d_size_out(state_dim[0], kernel_size=19, padding=1, stride=1)
        convh = conv2d_size_out(state_dim[0], kernel_size=19, padding=1, stride=1)

        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, 128)
        self.out = nn.Linear(128, action_num)

    def forward(self, state):
        obs = state.permute(0, 3, 1, 2)
        x = t.selu(self.conv1(obs))
        x = t.selu(self.head(x.reshape(x.size(0), -1)))
        return self.out(x)

if __name__ == "__main__":
    device = t.device('cpu')
    q_net = deepmind_Qnet(observe_dim, action_num).to(device)
    q_net_t = deepmind_Qnet(observe_dim, action_num).to(device)

    dqn_per = DQNPer(q_net, q_net_t, t.optim.Adam, nn.MSELoss(reduction="sum"), discount=0.95, epsilon_decay=0.998,
                     update_rate=0.05, replay_buffer=PrioritizedBuffer(500000, "cpu", beta=0.4))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    first_reward = False

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset()["image"], dtype=t.float32).view(1, *observe_dim)
        tmp_observations = []

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = dqn_per.act_discrete_with_noise({"state": old_state})
                state, reward, terminal, _ = env.step(action.item())
                if reward > 0:
                    print(step, reward)
                state = t.tensor(state["image"], dtype=t.float32).view(1, *observe_dim)
                total_reward += reward

                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )

        dqn_per.store_episode(tmp_observations)
        for _ in range(60):
            dqn_per.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0
