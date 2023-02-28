from data import *
from model import *
from train import *
import random

seed = 42

ROOT = Path(__file__).parents[1].resolve()
tng_base_path = f"{ROOT}/illustris_data/TNG50-1/output"

experiment = "predict_Mhalo" # mass accretion rate, versus smhm

verbose = True
plot_figures = True

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    df = load_data(
        tng_base_path=tng_base_path,
        snapshot=config_params["snapshot"], # default is 99 -> z=0 
        use_stellarhalfmassradius=feature_params["use_stellarhalfmassradius"],
        use_velocity=feature_params["use_velocity"],
        use_only_positions=feature_params["use_only_positions"],
        in_projection=feature_params["in_projection"]
    )

    dataset, n_subhalos = generate_dataset(
        df,
        use_velocity=feature_params["use_velocity"],
        use_central_galaxy_frame=feature_params["use_central_galaxy_frame"],
        use_only_positions=feature_params["use_only_positions"],
        in_projection=feature_params["in_projection"]
    )
        
    node_features = dataset[0].x.shape[1]
    n_halos = len(dataset)
    
    print("Number of features:", node_features)
    print("Number of subhalos:", n_subhalos)
    print("Number of halos:", n_halos)

    # split dataset
    train_loader, valid_loader, test_loader = split_datasets(
        dataset,
        rng=rng,
        valid_frac=training_params["valid_frac"],
        test_frac=training_params["test_frac"],
        batch_size=training_params["batch_size"],
    )

    # load model
    model = EdgePointGNN(
        node_features=node_features, 
        n_layers=model_params["n_layers"], 
        k_nn=model_params["k_nn"],
        hidden_channels=model_params["n_hidden"],
        latent_channels=model_params["n_latent"],
        loop=model_params["loop"]
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_params["learning_rate"], 
        weight_decay=training_params["weight_decay"]
    )

    # training
    train_losses = []
    valid_losses = []
    for epoch in range(training_params["n_epochs"]):
        train_loss = train(train_loader, model, optimizer, device, in_projection=feature_params["in_projection"])
        valid_loss, valid_std, *_ = validate(valid_loader, model, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if verbose:
            print(f"Epoch {epoch + 1: >3d}    train loss: {train_loss: >4.1f}     valid loss: {valid_loss: >4.1f}     average valid std: {np.mean(valid_std): >5.3f}")
    
    test_loss, test_std, p_test, y_test, logvar_test = validate(test_loader, model, device)

    print(f"Test RMSE: {np.sqrt(np.mean((y_test - p_test)**2)): >4.3f}  Test loss: {test_loss: >4.1f} Test std: {test_std: >5.3f}")

    # save results
    np.save(f"{ROOT}/results/{experiment}/test_preds.npy", p_test)
    np.save(f"{ROOT}/results/{experiment}/test_trues.npy", y_test)
    np.save(f"{ROOT}/results/{experiment}/test_logvars.npy", logvar_test)

    if plot_figures:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4), dpi=200)

        plt.scatter(p_test+10, y_test+10, s=5, edgecolors='none', c="#003f5c")
        plt.xlabel("$\\log(M_\\bigstar/[M_\\odot\ h^{-1}])$")
        plt.ylabel("$\\log(M_{\\rm halo}/[M_\\odot\ h^{-1}])$")
        plt.xlim(10, 14)
        plt.ylim(10, 14)
        plt.grid(alpha=0.15)
        plt.savefig(f"{ROOT}/results/{experiment}/results.png")