"""
checkpoint.py
--------------
Utility functions for saving and loading model checkpoints consistently.
"""

import os
import torch

def save_checkpoint(model, optimizer=None, epoch=None, loss=None, name="model", path="../checkpoints"):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"{name}_epoch{epoch if epoch else 'final'}.pt")

    state = {"model_state": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if loss is not None:
        state["loss"] = loss

    torch.save(state, save_path)
    print(f"Saved checkpoint: {save_path}")

def load_checkpoint(model, optimizer=None, path=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Loaded model weights from {path}")
    return model
