import os
import numpy as np
import torch as th
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import zipfile

import pickle

with open("dataset_images_train", 'rb') as fo:
    data_dict = pickle.load(fo, encoding='bytes')

k = len(np.unique(data_dict['target']))
d = data_dict['data'].shape[1]

x = data_dict['data']
y = data_dict['target']

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = (x_train / 255.0 - 0.5) / 0.5
x_valid = (x_valid / 255.0 - 0.5) / 0.5

with open("data_images_test", 'rb') as fo:
    dict_test = pickle.load(fo, encoding='bytes')

x_test = dict_test['data']

def create_unique_folder(base_path="predictions", base_name="prediction"):
    counter = 1
    folder_name = f"{base_name}_{counter}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    folder_path = os.path.join(base_path, folder_name)
    while os.path.exists(folder_path):
        counter += 1
        folder_name = f"{base_name}_{counter}"
        folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path)
    return folder_path

class myDataset(th.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def prediction(f):
    return th.argmax(f, 1)

def error_rate(y_pred, y):
    return ((y_pred != y).sum().float()) / y_pred.size()[0]

class Reg_log_multi(th.nn.Module):
    # Constructeur qui initialise le modèle
    def __init__(self, params):
        super(Reg_log_multi, self).__init__()

        self.layer = th.nn.Linear(d,k)
        self.bn = th.nn.BatchNorm1d(k)
        self.dropout = th.nn.Dropout(params["dropout_rate"])
        self.layer.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    # Implémentation de la passe forward du modèle
    def forward(self, x):
        out = self.layer(x)
        out = self.bn(out)
        out = self.dropout(out)
        return out

params = {
    "eta": 0.01,
    "batch_size": 64,
    "patience": 10,
    "num_epochs": 3000,
    "dropout_rate": 0.1,
    "weight_decay": 0.0001,
    "momentum": 0.3,
}

param_grid_group1 = {
    "batch_size": [128],
    "eta": [0.0001]
}

param_grid_group2 = {
    "dropout_rate": [0, 0.1, 0.2],
    "weight_decay": [0, 1e-3, 1e-4],
    "momentum": [0, 0.1, 0.125, 0.15, 0.175, 0.2]
}

param_grid_group1 = [dict(zip(param_grid_group1.keys(), values)) for values in itertools.product(*param_grid_group1.values())]
param_grid_group2 = [dict(zip(param_grid_group2.keys(), values)) for values in itertools.product(*param_grid_group2.values())]
# param_grid_group3 = [dict(zip(param_grid_group3.keys(), values)) for values in itertools.product(*param_grid_group3.values())]

x_train = th.from_numpy(x_train).float()
y_train = th.from_numpy(y_train).long()

x_valid = th.from_numpy(x_valid).float()
y_valid = th.from_numpy(y_valid).long()

device = "cuda" if th.cuda.is_available() else "cpu"
print(f"Utilisation de : {device}")

def train_and_evaluate_model(params, folder_name):
    trainloader = th.utils.data.DataLoader(myDataset(x_train, y_train), batch_size=params["batch_size"], shuffle=True)
    validloader = th.utils.data.DataLoader(myDataset(x_valid, y_valid), batch_size=params["batch_size"], shuffle=False)

    model = Reg_log_multi(params).to(device)

    criterion = th.nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=params["eta"], weight_decay=params["weight_decay"], momentum=params["momentum"])
    
    best_test_loss = float('inf')
    best_epoch = 0
    counter = 0
    x = [0]

    y_loss_train = []
    y_loss_test = []
    y_accuracy_train = []
    y_accuracy_test = []

    pbar = tqdm(range(params["num_epochs"]))
    for epoch in pbar:
        train_loss = 0.0
        error_train = 0.0

        for data, target in trainloader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            y_pred_train = prediction(output)
            error_train += error_rate(y_pred_train, target)

        train_loss /= len(trainloader)
        error_train /= len(trainloader)
        accuracy_train = 1 - error_train

        y_loss_train.append(train_loss)
        y_accuracy_train.append(accuracy_train)

        test_loss = 0.0
        error_test = 0.0
        with th.no_grad():
            for data, target in validloader:
                data, target = data.to(device), target.to(device)

                output = model(data)

                loss = criterion(output, target)

                test_loss += loss.item()
                y_pred_test = prediction(output)
                error_test += error_rate(y_pred_test, target)

        test_loss /= len(validloader)
        error_test /= len(validloader)
        accuracy_test = 1 - error_test

        y_loss_test.append(test_loss)
        y_accuracy_test.append(accuracy_test)

        pbar.set_postfix(epoch=epoch+1, train_loss=train_loss, test_loss=test_loss, accuracy_test=accuracy_test)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            counter = 0

            th.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, os.path.join(folder_name, "best_model.pth"))
        else:
            counter += 1
            if counter >= params["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        x.append(epoch + 1)

    y_loss_train = [tensor.cpu().numpy() if isinstance(tensor, th.Tensor) else tensor for tensor in y_loss_train]
    y_loss_test = [tensor.cpu().numpy() if isinstance(tensor, th.Tensor) else tensor for tensor in y_loss_test]
    y_accuracy_train = [tensor.cpu().numpy() if isinstance(tensor, th.Tensor) else tensor for tensor in y_accuracy_train]
    y_accuracy_test = [tensor.cpu().numpy() if isinstance(tensor, th.Tensor) else tensor for tensor in y_accuracy_test]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x[:len(y_loss_train)], y_loss_train, c='r', label='Loss (train)')
    ax1.plot(x[:len(y_loss_test)], y_loss_test, c='b', label='Loss (valid)')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Erreur d'entraînement et de validation")
    ax1.legend()

    ax2.plot(x[:len(y_accuracy_train)], y_accuracy_train, c='g', label='Accuracy (train)')
    ax2.plot(x[:len(y_accuracy_test)], y_accuracy_test, c='orange', label='Accuracy (valid)')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy d'entraînement et de validation")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "training_validation_metrics.png"))
    plt.close()
    
    history_entry = {
        'best_epoch': best_epoch + 1,
        'best_test_loss': best_test_loss,
        'final_train_loss': train_loss,
        'final_test_loss': test_loss,
        'final_accuracy_train': y_accuracy_train[-1] if y_accuracy_train else None,
        'final_accuracy_test': y_accuracy_test[-1] if y_accuracy_test else None,
        'num_epochs': len(x),
        'eta': params["eta"],
        'batch_size': params["batch_size"],
        'dropout_rate': params["dropout_rate"],
        'weight_decay': params["weight_decay"],
        'momentum': params["momentum"]
    }
    
    history_file = "training_history.csv"

    write_header = not os.path.exists(history_file)
    
    with open(history_file, 'a') as f:
        if write_header:
            f.write(','.join(history_entry.keys()) + '\n')
        f.write(','.join(str(v) for v in history_entry.values()) + '\n')

    return best_test_loss

best_params_group1 = None
best_test_loss_group1 = float('inf')
for i, current_params in enumerate(param_grid_group1):
    print(f"Test de la combinaison {i + 1}/{len(param_grid_group1)} : {current_params}")
    params.update(current_params)
    folder_name = create_unique_folder(base_path="grid_search_results", base_name="grid_search_group1")
    test_loss = train_and_evaluate_model(params, folder_name)
    if test_loss < best_test_loss_group1:
        best_test_loss_group1 = test_loss
        best_params_group1 = current_params.copy()

best_params_group2 = None
best_test_loss_group2 = float('inf')
for i, current_params in enumerate(param_grid_group2):
    print(f"Test de la combinaison {i + 1}/{len(param_grid_group2)} : {current_params}")
    params.update(current_params)
    folder_name = create_unique_folder(base_path="grid_search_results", base_name="grid_search_group2")
    test_loss = train_and_evaluate_model(params, folder_name)
    if test_loss < best_test_loss_group2:
        best_test_loss_group2 = test_loss
        best_params_group2 = current_params.copy()

# best_params_group3 = None
# best_test_loss_group3 = float('inf')
# for i, current_params in enumerate(param_grid_group3):
#     print(f"Test de la combinaison {i + 1}/{len(param_grid_group3)} : {current_params}")
#     params.update(current_params)
#     folder_name = create_unique_folder(base_path="grid_search_results", base_name="grid_search_group3")
#     test_loss = train_and_evaluate_model(params, folder_name)
#     if test_loss < best_test_loss_group3:
#         best_test_loss_group3 = test_loss
#         best_params_group3 = current_params.copy()

print(best_params_group1)
print(best_params_group2)
# print(best_params_group3)