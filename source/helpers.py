import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim
import pandas as pd

from typing import List, Tuple
from tqdm import tqdm
from utils import rolloutBuffer, tunableConfig

def collect_trajectories(
    n_steps: int, 
    env: gymnasium.Env, 
    policy_model: nn.Module, 
    device: torch.device, 
    seed: int,
    argmax_action: bool
):
    pbar = tqdm(
        total=n_steps, 
        desc="collecting trajectories", 
        colour="green", 
        leave=False
    )

    observation, _ = env.reset(seed=seed)
    observation_tensor = torch.tensor(
        observation, 
        dtype=torch.float32, 
        device=device
    )

    rollouts: List[rolloutBuffer] = []
    completed_episode_rewards = []
    current_episode_reward = 0.0

    with torch.inference_mode(): 
        for _ in range(n_steps):
            # policy distribution & value estimate
            logits, value = policy_model(observation_tensor)
            action_dist = distributions.Categorical(logits=logits)
            
            # argmax action for validation
            if argmax_action:
                action_tensor = torch.argmax(logits)
            else:
                action_tensor = action_dist.sample()
            
            # perform action in environment
            action = action_tensor.item()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward_tensor = torch.tensor(
                reward, 
                dtype=torch.float32, 
                device=device
            )
            done_tensor = torch.tensor(
                done, 
                dtype=torch.bool, 
                device=device
            )

            current_episode_reward += reward

            # gather rollout buffer
            buffer = rolloutBuffer(
                observation=observation_tensor, 
                reward=reward_tensor,
                action=action_tensor,
                log_prob=action_dist.log_prob(action_tensor),
                value=value.squeeze(-1),
                entropy=action_dist.entropy(),
                mean=action_dist.mean,
                std=action_dist.stddev,
                perplexity=action_dist.perplexity(),
                done=done_tensor     
            )
            rollouts.append(buffer)

            if done:
                completed_episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                next_observation, _ = env.reset()

            observation_tensor = torch.tensor(
                next_observation, 
                dtype=torch.float32, 
                device=device
            )

            pbar.update()
            
    pbar.close()
    
    completed_episode_rewards = torch.tensor(completed_episode_rewards)
    return rollouts, completed_episode_rewards.mean(), completed_episode_rewards.std()

def compute_ppo_advantages(
    rollouts: List[rolloutBuffer], 
    gamma: float = 0.99
) -> Tuple[
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor, 
    torch.Tensor
]:
    returns = []
    R = 0.0
    for step in reversed(rollouts):
        if step.done:
            R = 0.0 
        R = step.reward.item() + gamma * R
        returns.insert(0, R)
        
    returns_tensor = torch.tensor(
        returns, 
        dtype=torch.float32, 
        device=rollouts[0].reward.device
    )
    values_tensor = torch.stack([r.value for r in rollouts])
    
    advantages = returns_tensor - values_tensor
    advantages_mean = advantages.mean()
    advantages_std = advantages.std()
    advantages_median = advantages.median()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return returns_tensor, advantages, advantages_mean, advantages_std, torch.tensor(advantages_median.item())

def update_ppo(
    epochs: int,
    policy_model: nn.Module, 
    optimiser: torch.optim.Optimizer, 
    rollouts: List[rolloutBuffer], 
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
    critic_loss_weight: float,
    entropy_regulariser_weight: float
) -> float:

    obs = torch.stack([r.observation for r in rollouts])
    actions = torch.stack([r.action for r in rollouts])
    old_log_probs = torch.stack([r.log_prob for r in rollouts]).detach()
    
    total_loss = 0.0
    
    pbar = tqdm(
        total=epochs,
        desc="ppo training",
        colour="yellow",
        leave=False
    )
    for _ in range(epochs):
        logits, values = policy_model(obs)
        dist = distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.smooth_l1_loss(values.squeeze(-1), returns)
        loss = actor_loss + critic_loss_weight * critic_loss - entropy_regulariser_weight * entropy

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item()

        pbar.update()
        pbar.set_postfix({
            "loss" : loss.item()
        })

    return total_loss / epochs

def __get_data(
    rollouts: List[rolloutBuffer],
    advantages_mean: torch.Tensor,
    advantages_median: torch.Tensor,
    advantages_std: torch.Tensor,
    total_reward_mean: torch.Tensor,
    total_reward_std: torch.Tensor,
    loss: float
):
    entropies = torch.stack([r.entropy for r in rollouts])
    perplexities = torch.stack([r.perplexity for r in rollouts])
    action_mean = torch.stack([r.mean for r in rollouts])
    action_std = torch.stack([r.std for r in rollouts])

    return {
        "total_reward_mean" : total_reward_mean.item(),
        "total_reward_std": total_reward_std.item(),

        "entropies_mean" : entropies.mean().item(),
        "entropies_std" : entropies.std().item(),
        "entropies_median" : entropies.median().item(),

        "advantages_mean" : advantages_mean.item(),
        "advantages_std" : advantages_std.item(),
        "advantages_median" : advantages_median.item(),

        "perplexities_mean" : perplexities.mean().item(),
        "perplexities_std" : perplexities.std().item(),
        "perplexities_median" : perplexities.median().item(),

        "action_mean_median" : action_mean.median().item(),
        "action_std_median" : action_std.median().item(),

        "loss": loss
    }

def train(
   config: tunableConfig,
   epochs: int,
   n_steps: int,
   env: gymnasium.Env,
   policy_model: nn.Module,
   optimiser: optim.Optimizer,
   desc: str,
   colour: str,
   train_analytics_savefile: str,
   eval_analytics_savefile: str,
   eval_argmax_analytics_savefile: str,
   policy_model_savefile: str,
   eval_only: bool
):
    pbar = tqdm(
        total=epochs, 
        desc=desc, 
        colour=colour
    )

    train_data = []
    eval_data = []
    eval_argmax_data = []
    
    for _ in range(epochs):
        if not eval_only:
            policy_model.train()
            # =----- collect trajectories & do ppo update -----=
            rollouts, total_reward_mean, total_reward_std = collect_trajectories(
                n_steps=n_steps, 
                env=env, 
                policy_model=policy_model, 
                device=config.device, 
                seed=config.seed,
                argmax_action=False
            )
            returns, advantages, advantages_mean, advantages_std, advantages_median = compute_ppo_advantages(
                rollouts=rollouts, 
                gamma=config.gamma
            )
            loss = update_ppo(
                epochs=config.epochs_ppo,
                policy_model=policy_model,
                optimiser=optimiser,
                rollouts=rollouts,
                returns=returns,
                advantages=advantages,
                clip_ratio=config.clip_ratio,
                critic_loss_weight=config.critic_loss_weight,
                entropy_regulariser_weight=config.entropy_regulariser_weight
            )
            
            train_data.append(__get_data(
                rollouts=rollouts,
                advantages_mean=advantages_mean,
                advantages_median=advantages_median,
                advantages_std=advantages_std,
                total_reward_mean=total_reward_mean,
                total_reward_std=total_reward_std,
                loss=loss
            ))

        # =----- random sampling evaluation -----=
        policy_model.eval()
        with torch.inference_mode():
            rollouts, total_reward_mean, total_reward_std = collect_trajectories(
                n_steps=n_steps, 
                env=env, 
                policy_model=policy_model, 
                device=config.device, 
                seed=config.seed,
                argmax_action=False
            )
            returns, advantages, advantages_mean, advantages_std, advantages_median = compute_ppo_advantages(
                rollouts=rollouts, 
                gamma=config.gamma
            )
            eval_data.append(__get_data(
                rollouts=rollouts,
                advantages_mean=advantages_mean,
                advantages_median=advantages_median,
                advantages_std=advantages_std,
                total_reward_mean=total_reward_mean,
                total_reward_std=total_reward_std,
                loss=loss
            ))

        # =----- argmax sampling evaluation -----=
        with torch.inference_mode():
            rollouts, total_reward_mean, total_reward_std = collect_trajectories(
                n_steps=n_steps, 
                env=env, 
                policy_model=policy_model, 
                device=config.device, 
                seed=config.seed,
                argmax_action=True
            )
            returns, advantages, advantages_mean, advantages_std, advantages_median = compute_ppo_advantages(
                rollouts=rollouts, 
                gamma=config.gamma
            )
            eval_argmax_data.append(__get_data(
                rollouts=rollouts,
                advantages_mean=advantages_mean,
                advantages_median=advantages_median,
                advantages_std=advantages_std,
                total_reward_mean=total_reward_mean,
                total_reward_std=total_reward_std,
                loss=loss
            ))

        pbar.update()

    df = pd.DataFrame(train_data)
    df.to_csv(train_analytics_savefile)
    df = pd.DataFrame(eval_data)
    df.to_csv(eval_analytics_savefile)
    df = pd.DataFrame(eval_argmax_data)
    df.to_csv(eval_argmax_analytics_savefile)

    torch.save(
        policy_model.state_dict(), 
        policy_model_savefile
    )