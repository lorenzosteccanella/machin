from .a2c import *
from machin.frame.buffers.prioritized_buffer import PrioritizedBuffer


class A2C_SIL(A2C):
    """
    A2C_SIL framework.
    """

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        critic: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        criterion: Callable,
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple, Tuple] = None,
        lr_scheduler_kwargs: Tuple[Dict, Dict] = None,
        batch_size: int = 100,
        actor_update_times: int = 5,
        critic_update_times: int = 10,
        actor_learning_rate: float = 0.001,
        critic_learning_rate: float = 0.001,
        entropy_weight: float = None,
        value_weight: float = 0.5,
        gradient_max: float = np.inf,
        gae_lambda: float = 1.0,
        discount: float = 0.99,
        normalize_advantage: bool = True,
        replay_size: int = 500000,
        replay_device: Union[str, t.device] = "cpu",
        replay_buffer: Buffer = None,
        replay_buffer_sil: Buffer = None,
        sil_actor_loss_weight: float = 1,
        sil_value_loss_weight: float = 0.01,
        sil_update_times: int=4,
        sil_batch_size: int = 512,
        normalize_advantage_sil = True,
        visualize: bool = False,
        visualize_dir: str = "",
        **__
    ):
        """
        see A2C



        """

        super().__init__(
            actor,
            critic,
            optimizer,
            criterion,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            batch_size=batch_size,
            actor_update_times=actor_update_times,
            critic_update_times=critic_update_times,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            entropy_weight=entropy_weight,
            value_weight=value_weight,
            gradient_max=gradient_max,
            gae_lambda=gae_lambda,
            discount=discount,
            normalize_advantage=normalize_advantage,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=replay_buffer,
            visualize=visualize,
            visualize_dir=visualize_dir,
        )
        self.replay_buffer_sil = (
            PrioritizedBuffer(replay_size, replay_device)
            if replay_buffer_sil is None
            else replay_buffer_sil
        )
        self.sil_actor_loss_weight = sil_actor_loss_weight
        self.sil_value_loss_weight = sil_value_loss_weight
        self.sil_batch_size = sil_batch_size
        self.sil_update_times = sil_update_times
        self.normalize_advantage_sil = normalize_advantage_sil

    def update_sil(self, **__):
        """
        Update network weights by self imitation learning

        """
        sum_act_loss = 0
        sum_sil_value_loss = 0
        self.actor.train()
        self.critic.train()
        for _ in range(self.sil_update_times):
            # sample a batch
            batch_size, (state, action, target_value), index, is_weight, = self.replay_buffer_sil.sample_batch(
                self.sil_batch_size,
                sample_method="random_unique",
                concatenate=True,
                sample_attrs=["state", "action", "value"],
                additional_concat_custom_attrs=["value"],
            )

            __, action_log_prob, *_ = self._eval_act(state, action)

            action_log_prob = action_log_prob.view(batch_size, 1)

            # calculate value loss
            value = self._criticize(state).detach()

            # calculate SIL policy loss
            sil_advantage = target_value.type_as(action_log_prob) - value.type_as(action_log_prob)
            sil_advantage[sil_advantage < 0] = 0
            # # normalize sil advantage
            if self.normalize_advantage_sil:
                sil_advantage = (sil_advantage - sil_advantage.mean()) / (sil_advantage.std() + 1e-6)
            act_policy_loss = -(action_log_prob * sil_advantage) * self.sil_actor_loss_weight
            act_policy_loss = act_policy_loss * t.from_numpy(is_weight).view([batch_size, 1])

            act_policy_loss = act_policy_loss.mean()
            sum_act_loss += act_policy_loss.item()

            # Update actor network
            self.actor.zero_grad()
            self._backward(act_policy_loss)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_max)
            self.actor_optim.step()

            # calculate sil value loss
            value = self._criticize(state)
            sil_value = target_value.type_as(value) - value
            sil_value[sil_value < 0] = 0
            sil_value_loss = sil_value * t.from_numpy(is_weight).view([batch_size, 1])
            sil_value_loss = (sil_value_loss * self.sil_value_loss_weight)  # SUM CRITERIA

            abs_value_error = (
                t.sum(t.abs(value - target_value.type_as(value)), dim=1)
                    .flatten()
                    .detach()
                    .cpu()
                    .numpy()
            )
            self.replay_buffer_sil.update_priority(abs_value_error, index)

            # Update critic network
            self.critic.zero_grad()
            self._backward(sil_value_loss.sum())  # SUM CRITERIA
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_max)
            self.critic_optim.step()

        return (
            -sum_act_loss / self.actor_update_times,
            sum_sil_value_loss / self.critic_update_times,
        )

    def store_episode_sil(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.

        "value" and "gae" are automatically calculated.
        """
        episode[-1]["value"] = episode[-1]["reward"]

        # calculate value for each transition
        for i in reversed(range(1, len(episode))):
            episode[i - 1]["value"] = (
                episode[i]["value"] * self.discount + episode[i - 1]["reward"]
            )

        self.replay_buffer_sil.store_episode(
            episode,
            required_attrs=(
                "state",
                "action",
                "next_state",
                "reward",
                "value",
                "terminal",
            ),
        )
