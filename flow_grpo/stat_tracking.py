from typing import List, Optional, Union, Callable, Dict
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerPromptStatTracker:
    def __init__(self, global_std=True, use_history=False):
        self.global_std = global_std
        self.use_history = use_history
        self.stats = {}
        self.history_prompts = set()

    def multi_reward_aggregate(
            self,
            prompts : List[str],
            rewards : dict[str, np.ndarray | torch.Tensor],
            reward_weights : Optional[dict[str, float]] = {},
            aggregate_fn: Optional[Callable[[Dict[str, np.ndarray]], np.ndarray]] = None,
            store_result: bool = True,
        ) -> np.ndarray:
        """
            Aggregate multi-dimensional rewards for each prompt group and optionally store the result.
            
            This function groups rewards by prompt and applies weighted aggregation within each group.
            It's typically used before `compute_advantages()` to combine multiple reward signals into
            a single reward value for advantage computation in RLHF training.
            
            Args:
                prompts: List of prompt strings. Samples with the same prompt are grouped together.
                rewards: Dictionary mapping reward names to reward arrays. Each array has shape 
                        (num_samples,) or (num_samples, num_timesteps) for temporal rewards.
                reward_weights: Optional dictionary of weights for each reward type. Defaults to 1.0
                                for all rewards if not specified.
                aggregate_fn: Optional aggregation function taking reward dict as kwargs and returning
                            aggregated array. Defaults to weighted sum across reward types.
                store_result: If True, adds aggregated result to rewards dict under 'avg' key.
            
            Returns:
                Aggregated reward array with same shape as input reward arrays.
            
            Example:
                >>> rewards = {'quality': np.array([0.8, 0.6]), 'safety': np.array([0.9, 0.7])}
                >>> weights = {'quality': 0.7, 'safety': 0.3}
                >>> agg_rewards = multi_reward_aggregate(['p1', 'p1'], rewards, weights)
                # Result: weighted sum per prompt group, then used in compute_advantages()
        """
        if aggregate_fn is None:
            # If not given, use np.sum directly
            aggregate_fn = lambda **kwargs: np.sum(list(kwargs.values()), axis=0)

        assert aggregate_fn is not None, "aggregate_fn must be provided for multi_reward_update."

        if reward_weights is None:
            reward_weights = {k: 1.0 for k in rewards.keys()}

        if 'avg' in rewards:
            # Drop 'avg' key if exists to avoid confusion and log a warning
            logger.warning("'avg' key found in rewards dictionary. It will be ignored in multi_reward_update.")
            rewards = {k: v for k, v in rewards.items() if k != 'avg'}
        
        # Aggregate rewards within each prompt (group)
        prompts = np.array(prompts)
        unique = np.unique(prompts)
        aggregated_rewards = np.zeros_like(next(iter(rewards.values())), dtype=np.float64)
        for prompt in unique:
            prompt_rewards = {k: np.array(v[prompts == prompt], dtype=np.float64) for k, v in rewards.items()}
            # Apply weights
            for k in prompt_rewards.keys():
                prompt_rewards[k] = prompt_rewards[k] * reward_weights.get(k, 1.0)
            # Aggregate
            aggregated = aggregate_fn(**prompt_rewards)
            aggregated_rewards[prompts == prompt] = aggregated
        
        # Store the aggregated rewards under 'avg' key
        if store_result:
            rewards['avg'] = aggregated_rewards

        return aggregated_rewards

    def compute_advantages(self, prompts : List[str], rewards : np.ndarray | torch.Tensor, type : str = 'grpo') -> np.ndarray:
        """
            Add `prompts` and corresponding `rewards` to the tracker and return advantages.

            rewards can be a tensor with extract timestep dimension for each prompt, of shape (prompt_num, timestep_num). Or just a one-dimensional array
            The return `advantage` keeps the same dimension as `rewards`.
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards, dtype=np.float64)

        # Group rewards by prompt
        for prompt in unique:
            # Get rewards for this prompt
            prompt_rewards = rewards[prompts == prompt]
            # Add rewards to self.stats
            if prompt not in self.stats:
                self.stats[prompt] = prompt_rewards
            else:
                self.stats[prompt] = np.concatenate([self.stats[prompt], prompt_rewards])

            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts

        # Compute mean and std for each sample
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]

            if type == 'rank-grpo':
                assert self.use_history == False, "Ranked-based GPRO does not support use_history=True"
                # Compute rank-based rewards for this prompt group
                prompt_rewards = self.compute_rank_rewards(prompt_rewards)

            # Compute mean and std
            if self.use_history:
                # 1. Use all its history when `use_history=True`
                mean_data = self.stats[prompt]
                if self.global_std:
                    # Global std across all history
                    std_data = np.concatenate(list(self.stats.values()))
                else:
                    # Local std across all history, for this prompt only
                    std_data = self.stats[prompt]
            else:
                # 2. Use only info in this update.
                mean_data = prompt_rewards
                if self.global_std:
                    # Global std across this update info
                    std_data = rewards
                else:
                    # Local std for this prompt only
                    std_data = prompt_rewards
    
            mean = np.mean(mean_data, axis=0, keepdims=True)
            std = np.std(std_data, axis=0, keepdims=True)

            # Avoid division by zero
            std = max(std, 1e-6)

            # Compute advantages with different algorithm
            if type == 'grpo' or type == 'rank-grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type == 'rwr':
                # advantages[prompts == prompt] = (prompt_rewards - mean) / std
                advantages[prompts == prompt] = prompt_rewards
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
            elif type == 'sft':
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
            elif type == 'dpo':
                # Get the advantages of the current prompt
                prompt_advantages = torch.tensor(prompt_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()
                # print("reward difference one group", prompt_advantages[max_idx]-prompt_advantages[min_idx])
            
        return advantages

    def compute_rank_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
            Compute rank-based rewards (related rewards) for a group of rewards (absolute rewards).
        """
        group_size = rewards.shape[0]
        ranked_rewards = np.argsort(np.argsort(rewards, axis=0), axis=0) / ((group_size - 1) if group_size > 1 else 1)
        return ranked_rewards

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        avg_group_std = np.mean([np.std(v) for v in self.stats.values()]) if self.stats else 0
        global_std = np.std(np.concatenate(list(self.stats.values()))) if self.stats else 0
        zero_std_ratio = sum(1 for v in self.stats.values() if np.std(v) < 1e-5) / len(self.stats) if self.stats else 0
        return avg_group_size, history_prompts, avg_group_std, global_std, zero_std_ratio

    def clear(self):
        self.stats = {}

def main():
    tracker = PerPromptStatTracker()

    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, -1, 4, 2, 1]
    advantages = tracker.compute_advantages(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.compute_advantages(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()