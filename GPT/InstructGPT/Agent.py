import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Agent(nn.Module):
    "The RL-based language model."
    def __init__(self,
                 model    # a pre-trained `transformers` model
    ):
        super().__init__()
        n_embd = model.config.n_embd
        self.eos_token_id = model.config.eos_token_id

        self.policy_network = model
        self.value_network = nn.Sequential(
            nn.Linear(n_embd, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

