import torch
import torch.nn as nn


class BinaryClassifierWithHidden(nn.Module):
    

    def __init__(self, input_dim=1024, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dims[1], 2)

    def load_legacy_state_dict(self, state_dict):
        mapped = {
            "fc1.weight": state_dict["network.0.weight"],
            "fc1.bias": state_dict["network.0.bias"],
            "fc2.weight": state_dict["network.3.weight"],
            "fc2.bias": state_dict["network.3.bias"],
            "out.weight": state_dict["network.6.weight"],
            "out.bias": state_dict["network.6.bias"],
        }
        return self.load_state_dict(mapped)

    def forward(self, x):
        h512 = self.act1(self.fc1(x))
        x = self.drop1(h512)
        h256 = self.act2(self.fc2(x))
        x = self.drop2(h256)
        logits = self.out(x)
        return {"h512": h512, "h256": h256, "logits": logits}


def load_classifier(path, device=None):
    classifier = BinaryClassifierWithHidden()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    classifier.load_legacy_state_dict(state)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)
    if device is not None:
        classifier.to(device)
    return classifier
