import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# Réseau de neurones avec des couches linéaires avec 4 couches
class Neural_network_multi_classif_4(th.nn.Module):
    # Constructeur qui initialise le modèle
    def __init__(self,d, k, params):
        super(Neural_network_multi_classif_4, self).__init__()
        
        self.layer1 = th.nn.Linear(d, params["h1"])
        self.layer2 = th.nn.Linear(params["h1"], params["h2"])
        self.layer3 = th.nn.Linear(params["h2"], params["h3"])
        self.layer4 = th.nn.Linear(params["h3"], params["h4"])
        self.layer5 = th.nn.Linear(params["h4"], k)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()
        self.layer4.reset_parameters()
        self.layer5.reset_parameters()

    # Implémentation de la passe forward du modèle
    def forward(self, x):
        phi1 = F.relu(self.layer1(x))
        phi2 = F.relu(self.layer2(phi1))
        phi3 = F.relu(self.layer3(phi2))
        phi4 = F.relu(self.layer4(phi3))

        return self.layer5(phi4)

# Réseau de neurones avec des couches linéaires avec 3 couches
class Neural_network_multi_classif_3(th.nn.Module):
    # Constructeur qui initialise le modèle
    def __init__(self, d, k, params):
        super(Neural_network_multi_classif_3, self).__init__()
        
        self.layer1 = th.nn.Linear(d, params["h1"])
        self.dropout1 = th.nn.Dropout(p=params["dropout_rate"])
        self.layer2 = th.nn.Linear(params["h1"], params["h2"])
        self.dropout2 = th.nn.Dropout(p=params["dropout_rate"])
        self.layer3 = th.nn.Linear(params["h2"], k)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

    # Implémentation de la passe forward du modèle
    def forward(self, x):
        phi1 = F.relu(self.layer1(x))
        phi1 = self.dropout1(phi1)
        phi2 = F.relu(self.layer2(phi1))
        phi2 = self.dropout2(phi2)

        return self.layer3(phi2)

def cnnet_3(d, k, params):
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Utilisation de : {device}")
    model = Neural_network_multi_classif_3(d, k, params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["eta"], weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, device, criterion, params

def cnnet_4(d, k, params):
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Utilisation de : {device}")
    model = Neural_network_multi_classif_4(d, k, params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["eta"], weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, device, criterion, params