from machin.frame.algorithms import A2C_SIL
from machin.utils.logging import default_logger as logger
from machin.frame.buffers.prioritized_buffer import PrioritizedBuffer
from torch.distributions import Categorical
import torch as t
import torch.nn as nn
import gym
import numpy as np
import random
from gym_minigrid.wrappers import NESWActionsImage

# configurations
env = gym.make("MiniGrid-DoorKey-6x6-v1")
env = NESWActionsImage(env, max_num_actions=60)
observe_dim = 4
action_num = 4
max_episodes = 200
max_steps = 60
solved_reward = 200
solved_repeat = 5

# Fixing all seed for reproducibility
seed = 1
np.random.seed(seed)
t.manual_seed(seed)
random.seed(seed)
env.seed(seed)

# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_num)

    def forward(self, state, action=None):
        a = t.selu(self.fc1(state))
        a = t.selu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        v = t.selu(self.fc1(state))
        v = t.selu(self.fc2(v))
        v = self.fc3(v)
        return v


if __name__ == "__main__":
    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)

    a2c_sil = A2C_SIL(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"), actor_learning_rate=0.0007,
                      critic_learning_rate=0.0007, entropy_weight=0.01, sil_update_times=4, sil_actor_loss_weight=1,
                      sil_value_loss_weight=0.01, normalize_advantage=False, actor_update_times=1, critic_update_times=2,
                      discount=0.95, replay_buffer_sil=PrioritizedBuffer(500000, "cpu", beta=0.1), value_weight=0.5,
                      gradient_max=1, sil_batch_size=256)

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset()["pos"], dtype=t.float32).view(1, observe_dim)

        tmp_observations = []
        tmp_observations_sil = []
        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = a2c_sil.act({"state": old_state})[0]
                state, reward, terminal, _ = env.step(action)
                state = t.tensor(state["pos"], dtype=t.float32).view(1, observe_dim)
                total_reward += reward
                #print((episode-1)*1000 + step, reward, smoothed_total_reward)
                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )

                tmp_observations_sil.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )

            if step % 40 == 0:
                a2c_sil.store_episode(tmp_observations)
                a2c_sil.update()
                a2c_sil.update_sil()
                tmp_observations.clear()


        a2c_sil.store_episode_sil(tmp_observations_sil)
        tmp_observations_sil.clear()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")
        if smoothed_total_reward >= solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0
