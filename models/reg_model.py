import torch as th
import torch.nn as nn
import torch.optim as optim

# Algorithme de régression logisitique multivariée
class Reg_log_multi(th.nn.Module):
    # Constructeur qui initialise le modèle
    def __init__(self, d, k, params):
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

def reg(d, k, params):
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Utilisation de : {device}")
    model = Reg_log_multi(d, k, params).to(device)
    optimizer = optim.SGD(model.parameters(), lr=params["eta"], weight_decay=params["weight_decay"], momentum=params["momentum"])
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, device, criterion, params