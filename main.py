import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg') # Pour sauvegarder dans des fichiers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import zipfile
from tqdm import tqdm

# Afficher les figures
plt_verbose = True

# 1.2 Importation des données du challenge
import pickle

with open("data/dataset_images_train", 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

# Dimenssions et classes
k = len(np.unique(dict['target']))
d = dict['data'].shape[1]

# 3.1 Visualisation des données

# 3.1.1 Affichage de la premiere image
# image = dict['data'][0].reshape(3, 32, 32)
# image = np.transpose(image, (1,2,0))
# plt.imshow(image)
# if plt_verbose:
#     plt.show()

# 3.1.2 Afficher n images d'une certaine classe k
def visualize_class(n, k):
    indices = np.where(dict['target'] == k)[0][:n]

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2, rows * 2))

    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')

        plt.imshow(np.transpose(dict['data'][idx].reshape(3, 32, 32), (1, 2, 0)))

    plt.savefig(f"visualize_class_{k}.png")

    if plt_verbose:
        plt.show()

    plt.close()

# visualize_class(10, 1) # Affiche 10 images de la classe k=1

def visualize_random_class(n):
    random_indices = np.random.choice(d, size=n, replace=False)  # Choix aléatoire des images
    
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2, rows * 2))

    for i, idx in enumerate(random_indices):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')

        plt.imshow(np.transpose(dict['data'][idx].reshape(3, 32, 32), (1, 2, 0)))
        plt.title(f"Label : {dict['target'][idx]}", fontsize=10)

    plt.savefig("visualize_random_class.png")

    if plt_verbose:
        plt.show()

    plt.close()

# visualize_random_class(9) # classe random

# 3.1.3 
# x = dict['data'][0:3000]
# y = dict['target'][0:3000]

# tsne = TSNE(n_components=2, random_state=42).fit_transform(x)

# fig = plt.scatter(tsne[:, 0], tsne[:, 1], c=y)
# plt.colorbar(fig)
# if plt_verbose:
#     plt.show()

# On remarque à vue d'oeil que toutes les classes sont bien représenté, et que certaines classes (ici la jaune) sont regroupées 

# 3.2 Première modélisation

# 3.2.1
x = dict['data']
y = dict['target']

# Création de 2 ensembles : apprentissage (80%) et validation (20%)
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalisation des ensembles pour optimiser les modélisations
from sklearn import preprocessing
x_train = (x_train / 255.0 - 0.5) / 0.5
x_valid = (x_valid / 255.0 - 0.5) / 0.5

# Algorithme des k plus proches voisins
from sklearn.neighbors import KNeighborsClassifier
def best_k(x_train, x_valid, y_train, y_valid, kmax = 20):
    list_accuracy = []
    for k in tqdm(range(1, kmax, 2)):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        list_accuracy.append(knn.score(x_valid, y_valid))

    return list_accuracy

# 3.2.2 & 3.2.3 
def visualize_best_k(x_train, x_valid, y_train, y_valid, kmax = 20, Verbose=True):
    list_accuracy = best_k(x_train, x_valid, y_train, y_valid, kmax)

    best_k_index = np.argmax(list_accuracy)

    best_k_value = range(1, kmax, 2)[best_k_index]
    best_accuracy = list_accuracy[best_k_index]

    plt.plot(range(1, kmax, 2), list_accuracy, marker='o', linestyle='-', color='b', zorder=0)
    plt.scatter(best_k_value, best_accuracy, color='red', s=75, label=f'Meilleur k = {best_k_value}', zorder=1)

    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title(f"Meilleur k pour l'algorithme des k plus proches voisins")
    plt.legend()

    plt.tight_layout()
    plt.savefig("best_k_KNeighborsClassifier.png")

    if plt_verbose:
        plt.show()

    plt.close()
    
    if Verbose:
        print(f"La meilleure valeur de k est : {best_k_value} avec une précision de {best_accuracy:.2f}")

# visualize_best_k(x_train, x_valid, y_train, y_valid, kmax=100, Verbose=False)

# 3.2.4
with open("data/data_images_test", 'rb') as fo:
    dict_test = pickle.load(fo, encoding='bytes')

x_test = dict_test['data']

# knn = KNeighborsClassifier(n_neighbors=1) # Le meilleur k était 1
# knn.fit(x, y)
# y_pred = knn.predict(x_test)

# np.savetxt("images_test_predictions.csv", y_pred)

# with zipfile.ZipFile("images_test_predictions.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
#     zipf.write("images_test_predictions.csv")

# 3.3 Autres modélisations plus avancées
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
import os

# Fonction pour archiver automatiquement nos résultats
def create_unique_folder(base_name="predicton"):
    counter = 1
    folder_name = f"{base_name}_{counter}"

    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    folder_path = os.path.join("predictions", folder_name)

    while os.path.exists(folder_path):
        counter += 1
        folder_name = f"{base_name}_{counter}"
        folder_path = os.path.join("predictions", folder_name)

    os.makedirs(folder_path)
    return folder_path

# Permet d'entrainer les modeles avec des batch (fournis)
class myDataset(th.utils.data.Dataset):

    def __init__(self, data, label):
        # Initialise dataset from source dataset
        self.data = data
        self.label  = label

    def __len__(self):
        # Return length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return one element of the dataset according to its index
        return self.data[idx], self.label[idx]

dataset_train = myDataset(x_train, y_train)
dataset_valid = myDataset(x_valid, y_valid)

def prediction(f):
    return th.argmax(f, 1)

def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]

# Hyper paramètres pour le modèle reg
params_reg = {
    "eta": 0.0001,
    "batch_size": 128,
    "patience": 10,
    "num_epochs": 150,
    "weight_decay": 0,
    "dropout_rate": 0,
    "momentum": 0.125,
}

# Hyper paramètres pour le modèle cnnet_4
params_cnnet_4 = {
    "h1": 1024,
    "h2": 512,
    "h3": 256,
    "h4":128,
    "eta": 0.0001,
    "batch_size": 64,
    "patience": 10,
    "num_epochs": 100,
    "weight_decay": 0.0001,
}

# Hyper paramètres pour le modèle cnnet_3
params_cnnet_3 = {
    "h1": 512,
    "h2": 512,
    "eta": 0.001,
    "batch_size": 128,
    "patience": 5,
    "num_epochs": 20,
    "dropout_rate": 0.1,
    "weight_decay": 0.0001,
}

# Hyper paramètres pour le modèle cnn_3
params_cnn_3 = {
    "in_channels": 3,
    "out_channels1": 128,
    "out_channels2": 256,
    "out_channels3": 128,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "pool_kernel_size": 3,
    "pool_stride": 2,
    "h1": 512,
    "h2": 128,
    "dropout_rate": 0.2,
    "batch_size": 64,
    "eta": 0.001,
    "weight_decay": 0.00001,
    "num_epochs": 50,
    "patience": 5,
}

# Hyper paramètres pour le modèle cnn_4
params_cnn_4 = {
    "in_channels": 3,
    "out_channels1": 128,
    "out_channels2": 128,
    "out_channels3": 256,
    "out_channels4": 512,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "pool_kernel_size": 2,
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

# Selectionner le modèle en décommentant la ligne associée
from models.reg_model import reg
from models.cnnet_model import cnnet_3, cnnet_4
from models.cnn_model import cnn_3, cnn_4

# model, optimizer, device, criterion, params = reg(d, k, params_reg)
# model, optimizer, device, criterion, params = cnnet_3(d, k, params_cnnet_3)
# model, optimizer, device, criterion, params = cnnet_4(d, k, params_cnnet_4)
# model, optimizer, device, criterion, x_train, x_valid, x_test, params = cnn_3(d, k, params_cnn_3, x_train, x_valid, x_test)
# model, optimizer, device, criterion, x_train, x_valid, x_test, params = cnn_4(d, k, params_cnn_4, x_train, x_valid, x_test)

x_train = th.from_numpy(x_train).float().to(device)
y_train = th.from_numpy(y_train).long().to(device)

x_valid = th.from_numpy(x_valid).float().to(device)
y_valid = th.from_numpy(y_valid).long().to(device)

trainloader = th.utils.data.DataLoader(myDataset(x_train, y_train), batch_size=params["batch_size"], shuffle=True)
validloader = th.utils.data.DataLoader(myDataset(x_valid, y_valid), batch_size=params["batch_size"], shuffle=False)

folder_name = create_unique_folder()

results = []

best_test_loss = float('inf')
best_epoch = 0
counter = 0

x = [0]
y_loss_train = []
y_loss_test = []
y_accuracy_train = []
y_accuracy_test = []

pbar = tqdm(range(params["num_epochs"]))
best_test_loss = float('inf')
best_epoch = 0
counter = 0

# Pour entrainer le modèle sans batch (déconseillé)
# for epoch in pbar:                
#     f_train = model(x_train)
    
#     loss = criterion(f_train, y_train)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     train_loss = loss.item()
#     y_pred_train = prediction(f_train)
#     error_train = error_rate(y_pred_train, y_train)
#     accuracy_train = 1 - error_train

#     y_loss_train.append(train_loss)
#     y_accuracy_train.append(accuracy_train)
    
#     with th.no_grad():
#         f_test = model(x_valid)
#         loss = criterion(f_test, y_valid)

#         test_loss = loss.item()
#         y_pred_test = prediction(f_test)
#         error_test = error_rate(y_pred_test, y_valid)
#         accuracy_test = 1 - error_test
        
#         y_loss_test.append(test_loss)
#         y_accuracy_test.append(accuracy_test)

#     pbar.set_postfix(iter=epoch, loss=train_loss, error_train=error_train, error_test=error_test, accuracy_test=accuracy_test)

#     if test_loss < best_test_loss:
#         best_test_loss = test_loss
#         best_epoch = epoch
#         counter = 0

#         # On sauvegarde à ce niveau la meilleur configuration des poids du modèle
#         th.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': test_loss,
#         }, os.path.join(folder_name, "best_model.pth"))
#     else:
#         counter += 1
#         if counter >= params["patience"]:
#             print(f"Early stopping at epoch {epoch+1}")
#             break

#     x.append(epoch + 1)

# Avec batch
for epoch in pbar:
    train_loss = 0.0
    error_train = 0.0

    for data, target in trainloader:
        data, target = data.to(device), target.to(device)

        f_train = model(data)
        
        loss = criterion(f_train, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_pred_train = prediction(f_train)
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

            f_test = model(data)
            loss = criterion(f_test, target)

            test_loss += loss.item()
            y_pred_test = prediction(f_test)
            error_test += error_rate(y_pred_test, target)

    test_loss /= len(validloader)
    error_test /= len(validloader)
    accuracy_test = 1 - error_test
    y_loss_test.append(test_loss)
    y_accuracy_test.append(accuracy_test)

    pbar.set_postfix(iter=epoch, loss=train_loss, error_train=error_train, error_test=error_test, accuracy_test=accuracy_test)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch
        counter = 0

        # On sauvegarde à ce niveau la meilleur configuration des poids du modèle dans best_model.pth
        th.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
        }, os.path.join(folder_name, "best_model.pth"))
    else:
        counter += 1
        if counter >= params["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

    x.append(epoch + 1)

# Pour le support de cuda (matplotlib prend que des tenseurs en cpu)
y_loss_train = [tensor.cpu().numpy() if isinstance(tensor, th.Tensor) else tensor for tensor in y_loss_train]
y_loss_test = [tensor.cpu().numpy() if isinstance(tensor, th.Tensor) else tensor for tensor in y_loss_test]
y_accuracy_train = [tensor.cpu().numpy() if isinstance(tensor, th.Tensor) else tensor for tensor in y_accuracy_train]
y_accuracy_test = [tensor.cpu().numpy() if isinstance(tensor, th.Tensor) else tensor for tensor in y_accuracy_test]

fig_file = os.path.join(folder_name, "training_validation_metrics.png")

# Génere le fichier de sortie des metrics du modèle entrainé ci-dessus
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(x[:len(y_loss_train)], y_loss_train, c='r', label='Loss (train)')
ax1.plot(x[:len(y_loss_test)], y_loss_test, c='b', label='Loss (valid)')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.set_title("Loss d'entraînement et de validation")
ax1.legend()

ax2.plot(x[:len(y_accuracy_train)], y_accuracy_train, c='g', label='Accuracy (train)')
ax2.plot(x[:len(y_accuracy_test)], y_accuracy_test, c='orange', label='Accuracy (valid)')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy d'entraînement et de validation")
ax2.legend()

plt.tight_layout()
plt.savefig(fig_file)

if plt_verbose:
    plt.show()
    
plt.close()

# Fichier test déja importé au niveau du KNN
x_test = (x_test / 255.0 - 0.5) / 0.5
x_test = th.from_numpy(x_test).float().to(device)

checkpoint = th.load(os.path.join(folder_name, "best_model.pth"), map_location=device)
model_state_dict = checkpoint['model_state_dict']

model.load_state_dict(model_state_dict)

with th.no_grad():
    f_test = model(x_test)
    y_pred = prediction(f_test).cpu().numpy()

predictions_file = os.path.join(folder_name, "images_test_predictions.csv")
np.savetxt(predictions_file, y_pred)

params_file = os.path.join(folder_name, "params.txt")
with open(params_file, "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

import zipfile
with zipfile.ZipFile(os.path.join(folder_name, "images_test_predictions.zip"), 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(os.path.join(folder_name, "images_test_predictions.csv"), "images_test_predictions.csv")