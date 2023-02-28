import numpy as np
import torch
from scipy.spatial.transform import Rotation


training_params = dict(
    batch_size=128,
    learning_rate=3e-4,
    weight_decay=1e-8,
    n_epochs=100,
    valid_frac=0.15, # fraction of dataset for validation
    test_frac=0.15,  # fraction of dataset for testing
)

def train(dataloader, model, optimizer, device, in_projection=True):
    model.train()

    loss_total = 0
    for data in dataloader:

        # random rotation for data augmentation
        if in_projection:
            R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32)[:2, :2]
            data.pos = (R @ data.pos.unsqueeze(-1)).squeeze()
            data.x[:, :2] = (R @ data.x[:, :2].unsqueeze(-1)).squeeze()
        else:
            R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32)
            data.pos = (R @ data.pos.unsqueeze(-1)).squeeze()
            data.x[:, :3] = (R @ data.x[:, :3].unsqueeze(-1)).squeeze()

        data.to(device)

        optimizer.zero_grad()
        y_pred, logvar_pred = model(data).chunk(2, dim=1)

        # compute loss as sum of two terms for likelihood-free inference
        loss_mse = torch.nn.functional.mse_loss(y_pred.flatten(), data.y)
        loss_lfi = torch.nn.functional.mse_loss(y_pred.flatten() - data.y, 10**logvar_pred.flatten())
        loss = torch.log(loss_mse) + torch.log(loss_lfi)

        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    return loss_total / len(dataloader)

def validate(dataloader, model, device):
    model.eval()

    uncertainties = []
    loss_total = 0

    y_preds = []
    y_trues = []
    logvar_preds = []

    for data in dataloader:
        with torch.no_grad():
            data.to(device)
            y_pred, logvar_pred = model(data).chunk(2, dim=1)
            uncertainties.append(np.sqrt(10**logvar_pred.detach().cpu().numpy()).mean())

            # compute loss as sum of two terms for likelihood-free inference
            loss_mse = torch.nn.functional.mse_loss(y_pred.flatten(), data.y)
            loss_lfi = torch.nn.functional.mse_loss(y_pred.flatten() - data.y, 10**logvar_pred.flatten())
            loss = torch.log(loss_mse) + torch.log(loss_lfi)

            loss_total += loss.item()
            y_preds += list(y_pred.detach().cpu().numpy())
            y_trues += list(data.y.detach().cpu().numpy())
            logvar_preds += list(logvar_pred.detach().cpu().numpy())

    y_preds = np.concatenate(y_preds)
    y_trues = np.array(y_trues)
    logvar_preds = np.concatenate(logvar_preds)

    return (
        loss_total / len(dataloader),
        np.mean(uncertainties),
        y_preds,
        y_trues,
        logvar_preds
    )
    