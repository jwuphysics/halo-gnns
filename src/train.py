import numpy as np
import torch
from scipy.spatial.transform import Rotation
import torch.nn.functional as F

training_params = dict(
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=1e-5,
    n_epochs=200,
    valid_frac=0.15, # fraction of dataset for validation
    test_frac=0.15,  # fraction of dataset for testing
)

def train(dataloader, model, optimizer, device, augment=True, in_projection=False, no_positions=False, no_velocities=False):
    """Assumes that data object in dataloader has 8 columns: x,y,z, vx,vy,vz, Mh, Vmax"""
    model.train()

    loss_total = 0
    for data in dataloader:
        if augment: # add random noise
            data_node_features_scatter = 4e-3 * torch.randn_like(data.x) * torch.std(data.x, dim=0)
            data_edge_features_scatter = 4e-3 * torch.randn_like(data.edge_attr) * torch.std(data.edge_attr, dim=0)
            
            data.x += data_node_features_scatter
            data.edge_attr += data_edge_features_scatter

        data.to(device)

        optimizer.zero_grad()
        y_pred, logvar_pred = model(data).chunk(2, dim=1)
        y_pred = y_pred.view(-1, model.n_out)
        logvar_pred = logvar_pred.view(-1, model.n_out)

        # compute loss as sum of two terms for likelihood-free inference
        loss_mse = F.mse_loss(y_pred, data.y)
        loss_lfi = F.mse_loss((y_pred - data.y)**2, 10**logvar_pred)

        loss = torch.log(loss_mse) + torch.log(loss_lfi)

        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    return loss_total / len(dataloader)

def validate(dataloader, model, device, in_projection=False, no_velocities=False, no_positions=False):
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
            y_pred = y_pred.view(-1, model.n_out)
            logvar_pred = logvar_pred.view(-1, model.n_out)
            uncertainties.append(np.sqrt(10**logvar_pred.detach().cpu().numpy()).mean(-1))

            # compute loss as sum of two terms a la Moment Networks (Jeffrey & Wandelt 2020)
            loss_mse = F.mse_loss(y_pred, data.y)
            loss_lfi = F.mse_loss((y_pred - data.y)**2, 10**logvar_pred)

            loss = torch.log(loss_mse) + torch.log(loss_lfi)

            loss_total += loss.item()
            y_preds += list(y_pred.detach().cpu().numpy())
            y_trues += list(data.y.detach().cpu().numpy())
            logvar_preds += list(logvar_pred.detach().cpu().numpy())

    y_preds = np.concatenate(y_preds)
    y_trues = np.array(y_trues)
    logvar_preds = np.concatenate(logvar_preds)
    uncertainties = np.concatenate(uncertainties)

    return (
        loss_total / len(dataloader),
        np.mean(uncertainties, -1),
        y_preds,
        y_trues,
        logvar_preds
    )
    