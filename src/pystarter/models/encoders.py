import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x