from operator import truediv
from pty import STDIN_FILENO
from typing import List
import numpy as np
import torch

class PerPromptStatTracker:
    def __init__(self, global_std=True, use_history=False):
        self.global_std = global_std
        self.use_history = use_history
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts : List[str], rewards : List[float], type : str = 'grpo'):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0

        # Group rewards by prompt
        for prompt in unique:
            # Get rewards for this prompt
            prompt_rewards = rewards[prompts == prompt]
            # Add rewards to self.stats
            if prompt not in self.stats:
                self.stats[prompt] = []

            self.stats[prompt] = np.concatenate([self.stats[prompt], prompt_rewards])
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts

        # Compute mean and std for each sample
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            # Compute mean and std
            if self.use_history:
                # 1. Use all its history when `use_history=True`
                mean = np.mean(self.stats[prompt], axis=0)
                if self.global_std:
                    # Global std across all history
                    std = np.std(np.concatenate(list(self.stats.values())), axis=0) + 1e-4
                else:
                    # Local std across all history, for this prompt only
                    std = np.std(self.stats[prompt], axis=0) + 1e-4
            else:
                # 2. Use only info in this update.
                mean = np.mean(prompt_rewards, axis=0)
                if self.global_std:
                    # Global std across this update info
                    std = np.std(rewards, axis=0) + 1e-4
                else:
                    # Local std for this prompt only
                    std = np.std(prompt_rewards, axis=0) + 1e-4

            # Compute advantages with different algorithm
            if type == 'grpo':
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

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}

def main():
    tracker = PerPromptStatTracker()

    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, -1, 4, 2, 1]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()