import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# Réseau de convolution
class Conv_neural_network_multi_classif_3(th.nn.Module):
    def __init__(self, d, k, params):
        super(Conv_neural_network_multi_classif_3, self).__init__()
        
        self.conv1 = th.nn.Conv2d(params["in_channels"], params["out_channels1"], params["kernel_size"], params["stride"], params["padding"])
        self.bn1 = th.nn.BatchNorm2d(params["out_channels1"])
        self.conv2 = th.nn.Conv2d(params["out_channels1"], params["out_channels2"], params["kernel_size"], params["stride"], params["padding"])
        self.bn2 = th.nn.BatchNorm2d(params["out_channels2"])
        self.conv3 = th.nn.Conv2d(params["out_channels2"], params["out_channels3"], params["kernel_size"], params["stride"], params["padding"])
        self.bn3 = th.nn.BatchNorm2d(params["out_channels3"])
        
        self.pool = th.nn.MaxPool2d(params["pool_kernel_size"], params["pool_stride"])
        
        self._to_linear = None
        self._get_conv_output()

        self.fc1 = th.nn.Linear(self._to_linear, params["h1"])
        self.fc2 = th.nn.Linear(params["h1"], params["h2"])
        self.fc3 = th.nn.Linear(params["h2"], k)
        
        self.dropout = th.nn.Dropout(params["dropout_rate"])
        
    def _get_conv_output(self):
        with th.no_grad():
            input = th.zeros(1, 3, 32, 32)
            output = self.pool(F.relu(self.conv1(input)))
            output = self.pool(F.relu(self.conv2(output)))
            output = self.pool(F.relu(self.conv3(output)))
            self._to_linear = output.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        return self.fc3(x)

class Conv_neural_network_multi_classif_4(th.nn.Module):
    def __init__(self,d ,k , params):
        super(Conv_neural_network_multi_classif_4, self).__init__()
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

def cnn_3(d, k, params, x_train, x_valid, x_test):
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Utilisation de : {device}")
    model = Conv_neural_network_multi_classif_3(d, k, params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["eta"], weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    x_train = x_train.reshape(-1, 3, 32, 32)
    x_valid = x_valid.reshape(-1, 3, 32, 32)
    x_test = x_test.reshape(-1, 3, 32, 32)

    return model, optimizer, device, criterion, x_train, x_valid, x_test, params

def cnn_4(d, k, params, x_train, x_valid, x_test):
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Utilisation de : {device}")
    model = Conv_neural_network_multi_classif_4(d, k, params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["eta"], weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    x_train = x_train.reshape(-1, 3, 32, 32)
    x_valid = x_valid.reshape(-1, 3, 32, 32)
    x_test = x_test.reshape(-1, 3, 32, 32)

    return model, optimizer, device, criterion, x_train, x_valid, x_test, params