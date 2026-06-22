
import torch
import torch.nn as nn
import torch.nn.init as init


class MLPSequential(nn.Module):
    """
    MLP construit avec nn.Sequential pour la classification binaire.
    Architecture : Input -> Linear(30, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> Sigmoid
    """
    def __init__(self, input_dim):
        super(MLPSequential, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)


class MLPClass(nn.Module):
    """
    MLP implémenté en tant que classe personnalisée héritant de nn.Module.
    Même architecture que MLPSequential.
    """
    def __init__(self, input_dim):
        super(MLPClass, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def inspect_model_parameters(model):
    """
    Inspecte les paramètres d'un modèle en utilisant named_parameters() et state_dict().
    
    Args:
        model (nn.Module): Modèle PyTorch à inspecter
    """
    print("\n=== Paramètres avec named_parameters() ===")
    for name, param in model.named_parameters():
        print(f"Nom : {name}, Forme : {param.shape}, Requiert grad : {param.requires_grad}")
    
    print("\n=== state_dict() keys ===")
    for key in model.state_dict().keys():
        print(f"Clé : {key}")


def initialize_weights(model, strategy='xavier'):
    """
    Applique une stratégie d'initialisation aux poids du modèle.
    
    Args:
        model (nn.Module): Modèle PyTorch à initialiser
        strategy (str): Stratégie d'initialisation ('gaussian', 'constant', 'xavier')
    """
    def init_fn(m):
        if isinstance(m, nn.Linear):
            if strategy == 'gaussian':
                # Initialisation gaussienne (moyenne 0, écart-type 0.01)
                init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif strategy == 'constant':
                # Initialisation constante (0.0 pour les poids, 0.1 pour les biais)
                init.constant_(m.weight, 0.0)
                if m.bias is not None:
                    init.constant_(m.bias, 0.1)
            elif strategy == 'xavier':
                # Initialisation Xavier (uniforme)
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            else:
                raise ValueError(f"Stratégie d'initialisation inconnue : {strategy}")
    
    model.apply(init_fn)


if __name__ == "__main__":
    # Test des modèles et des fonctions
    input_dim = 30
    model_seq = MLPSequential(input_dim)
    model_class = MLPClass(input_dim)
    
    print("=== MLPSequential ===")
    inspect_model_parameters(model_seq)
    print("\n=== MLPClass ===")
    inspect_model_parameters(model_class)
    
    # Test des initialisations
    print("\n=== Initialisation Xavier ===")
    initialize_weights(model_seq, 'xavier')
    for name, param in model_seq.named_parameters():
        if 'weight' in name:
            print(f"{name} : moyenne = {param.mean().item():.4f}, écart-type = {param.std().item():.4f}")
