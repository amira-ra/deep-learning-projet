
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    Architecture classique LeNet-5 pour classification d'images.
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # Couches convolutives
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.sigmoid1 = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.sigmoid2 = nn.Sigmoid()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Couches fully connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.sigmoid3 = nn.Sigmoid()
        self.fc2 = nn.Linear(120, 84)
        self.sigmoid4 = nn.Sigmoid()
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sigmoid2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid3(x)
        x = self.fc2(x)
        x = self.sigmoid4(x)
        x = self.fc3(x)
        return x


class LeNetImproved(nn.Module):
    """
    Variante améliorée de LeNet-5 avec ReLU, BatchNorm, Dropout et MaxPool.
    """
    def __init__(self, num_classes=10):
        super(LeNetImproved, self).__init__()
        # Couches convolutives
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Couches fully connected avec Dropout
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class SimpleMLP(nn.Module):
    """
    MLP simple pour comparaison avec le CNN.
    """
    def __init__(self, input_dim, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_feature_maps(model, x):
    """
    Exécute une image dans le modèle et retourne les feature maps après chaque couche conv.
    Args:
        model: modèle CNN
        x: tenseur d'entrée (1, C, H, W)
    Returns:
        list of feature maps (un tenseur par couche conv)
    """
    feature_maps = []
    # Itérer sur les modules enfants du modèle
    for name, layer in model.named_children():
        x = layer(x)
        # Si la couche est une Conv2d, ajouter ses feature maps à la liste
        if isinstance(layer, nn.Conv2d):
            feature_maps.append(x.clone().detach().cpu())
    return feature_maps


def count_parameters(model):
    """
    Compte le nombre total de paramètres entraînables dans le modèle.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Test des modèles ===")
    
    # Instancier les modèles
    lenet = LeNet(num_classes=10)
    lenet_improved = LeNetImproved(num_classes=10)
    # Pour MNIST: 28x28 pixels en niveaux de gris → input_dim = 28*28 = 784
    mlp = SimpleMLP(input_dim=28*28, num_classes=10)
    
    print("\n--- Architecture de LeNet ---")
    for name, module in lenet.named_modules():
        if name != '':  # Ne pas afficher le module racine lui-même
            print(f"{name}: {module}")
    print(f"Nombre de paramètres: {count_parameters(lenet)}")
    
    print("\n--- Architecture de LeNetImproved ---")
    for name, module in lenet_improved.named_modules():
        if name != '':
            print(f"{name}: {module}")
    print(f"Nombre de paramètres: {count_parameters(lenet_improved)}")
    
    print("\n--- Architecture de SimpleMLP ---")
    for name, module in mlp.named_modules():
        if name != '':
            print(f"{name}: {module}")
    print(f"Nombre de paramètres: {count_parameters(mlp)}")
    
    # Test forward pass et get_feature_maps
    print("\n--- Test forward pass et get_feature_maps ---")
    x_test = torch.randn(1, 1, 28, 28)  # Batch de 1 image MNIST
    lenet.eval()
    with torch.no_grad():
        output = lenet(x_test)
        fmaps = get_feature_maps(lenet, x_test)
    print(f"Forme de la sortie LeNet: {output.shape}")
    print(f"Nombre de feature maps extraites: {len(fmaps)}")
    for i, fmap in enumerate(fmaps):
        print(f"Feature map {i+1} (après conv{i+1}): {fmap.shape}")
