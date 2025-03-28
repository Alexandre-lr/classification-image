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

class Conv_neural_network_multi_classif(th.nn.Module):
    def __init__(self, params):
        super(Conv_neural_network_multi_classif, self).__init__()
        self.conv1 = th.nn.Conv2d(params["in_channels"], params["out_channels1"], params["kernel_size"], params["stride"], params["padding"])
        self.bn1 = th.nn.BatchNorm2d(params["out_channels1"])
        self.conv2 = th.nn.Conv2d(params["out_channels1"], params["out_channels2"], params["kernel_size"], params["stride"], params["padding"])
        self.bn2 = th.nn.BatchNorm2d(params["out_channels2"])
        self.conv3 = th.nn.Conv2d(params["out_channels2"], params["out_channels3"], params["kernel_size"], params["stride"], params["padding"])
        self.bn3 = th.nn.BatchNorm2d(params["out_channels3"])
        self.conv4 = th.nn.Conv2d(params["out_channels3"], params["out_channels4"], params["kernel_size"], params["stride"], params["padding"])
        self.bn4 = th.nn.BatchNorm2d(params["out_channels4"])

        self.pool = th.nn.MaxPool2d(params["pool_kernel_size"], params["pool_stride"])

        self._to_linear = None
        self._get_conv_output()

        self.fc1 = th.nn.Linear(self._to_linear, params["h1"])
        self.fc2 = th.nn.Linear(params["h1"], params["h2"])
        self.fc3 = th.nn.Linear(params["h2"], params["h3"])
        self.fc4 = th.nn.Linear(params["h3"], 10)

        self.dropout = th.nn.Dropout(params["dropout_rate"])

    def _get_conv_output(self):
        with th.no_grad():
            input = th.zeros(1, 3, 32, 32)

            output = self.pool(F.relu(self.bn1(self.conv1(input))))
            output = self.pool(F.relu(self.bn2(self.conv2(output))))
            output = self.pool(F.relu(self.bn3(self.conv3(output))))
            output = self.pool(F.relu(self.bn4(self.conv4(output))))

            self._to_linear = output.view(output.size(0), -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        return self.fc4(x)

# class Conv_neural_network_multi_classif(th.nn.Module):
#     def __init__(self, params):
#         super(Conv_neural_network_multi_classif, self).__init__()
#         self.conv1 = th.nn.Conv2d(params["in_channels"], params["out_channels1"], params["kernel_size"], params["stride"], params["padding"])
#         self.bn1 = th.nn.BatchNorm2d(params["out_channels1"])
#         self.conv2 = th.nn.Conv2d(params["out_channels1"], params["out_channels2"], params["kernel_size"], params["stride"], params["padding"])
#         self.bn2 = th.nn.BatchNorm2d(params["out_channels2"])
#         self.conv3 = th.nn.Conv2d(params["out_channels2"], params["out_channels3"], params["kernel_size"], params["stride"], params["padding"])
#         self.bn3 = th.nn.BatchNorm2d(params["out_channels3"])

#         self.pool = th.nn.MaxPool2d(params["pool_kernel_size"], params["pool_stride"])

#         self._to_linear = None
#         self._get_conv_output()

#         self.fc1 = th.nn.Linear(self._to_linear, params["h1"])
#         self.fc2 = th.nn.Linear(params["h1"], params["h2"])
#         self.fc3 = th.nn.Linear(params["h2"], 10)

#         self.dropout = th.nn.Dropout(params["dropout_rate"])

#     def _get_conv_output(self):
#         with th.no_grad():
#             input = th.zeros(1, 3, 32, 32)

#             output = self.pool(F.relu(self.bn1(self.conv1(input))))
#             output = self.pool(F.relu(self.bn2(self.conv2(output))))
#             output = self.pool(F.relu(self.bn3(self.conv3(output))))

#             self._to_linear = output.view(output.size(0), -1).size(1)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))

#         x = x.view(x.size(0), -1)

#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)

#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)

#         return self.fc3(x)

params = {
    "in_channels": 3,
    "out_channels1": 128,
    "out_channels2": 128,
    "out_channels3": 256,
    "out_channels4": 512,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "pool_kernel_size": 3,
    "pool_stride": 2,
    "h1": 512,
    "h2": 1024,
    "h3": 512,
    "dropout_rate": 0,
    "batch_size": 128,
    "eta": 0.001,
    "weight_decay": 0,
    "num_epochs": 50,
    "patience": 5,
}

param_grid_group1 = {
    "dropout_rate": [0, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3],
    "weight_decay": [0, 1e-3, 1e-4, 1e-5], 
    "pool_kernel_size": [2, 3],
}

param_grid_group1 = [dict(zip(param_grid_group1.keys(), values)) for values in itertools.product(*param_grid_group1.values())]

x_train = x_train.reshape(-1, 3, 32, 32)
x_valid = x_valid.reshape(-1, 3, 32, 32)

x_train = th.from_numpy(x_train).float()
y_train = th.from_numpy(y_train).long()

x_valid = th.from_numpy(x_valid).float()
y_valid = th.from_numpy(y_valid).long()

device = "cuda" if th.cuda.is_available() else "cpu"
print(f"Utilisation de : {device}")

def train_and_evaluate_model(params, folder_name):
    trainloader = th.utils.data.DataLoader(myDataset(x_train, y_train), batch_size=params["batch_size"], shuffle=True)
    validloader = th.utils.data.DataLoader(myDataset(x_valid, y_valid), batch_size=params["batch_size"], shuffle=False)

    model = Conv_neural_network_multi_classif(params).to(device)

    criterion = th.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=params["eta"], weight_decay=params["weight_decay"])

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
        'out_channels1': params["out_channels1"],
        'out_channels2': params["out_channels2"],
        'out_channels3': params["out_channels3"],
        'out_channels4': params["out_channels4"],
        'h1': params["h1"],
        'h2': params["h2"],
        'h3': params["h3"],
        'eta': params["eta"],
        'batch_size': params["batch_size"],
        'dropout_rate': params["dropout_rate"],
        'weight_decay': params["weight_decay"],
        'stride': params["stride"],
        'padding': params["padding"],
        'kernel_size': params["kernel_size"],
        'pool_kernel_size': params["pool_kernel_size"],
        'pool_stride': params["pool_stride"]
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

print(best_params_group1)