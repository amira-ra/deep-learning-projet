# Projet Deep Learning — Amira Rmaili

Projet de Deep Learning en PyTorch composé de trois parties indépendantes : un MLP pour la classification tabulaire, un CNN type LeNet pour la classification d'images, et des modèles séquentiels (modèle de langage et Seq2Seq).

## Structure du projet

```
deep_learning/
├── partie_1_mlp/        # Perceptron multicouche (MLP)
│   ├── data_prep.py     # Préparation des données (Breast Cancer Wisconsin)
│   ├── models.py        # Architectures MLP (Sequential et orientée classes)
│   ├── train.py         # Boucle d'entraînement et évaluation
│   └── main_mlp.py       # Script principal : entraîne plusieurs configurations
│
├── partie_2_cnn/         # Réseau convolutif (CNN)
│   ├── manual_ops.py     # Implémentation manuelle de la convolution et du pooling
│   ├── models.py         # LeNet, LeNet amélioré, MLP de référence
│   ├── train.py          # Boucle d'entraînement et évaluation
│   └── main_cnn.py        # Script principal : entraînement + étude d'ablation
│
├── partie_3_seq2seq/      # Modèles séquentiels
│   ├── data_prep.py      # Vocabulaire et préparation des séquences
│   ├── models.py          # Modèle de langage (RNN/LSTM/GRU), Encoder/Decoder, Seq2Seq
│   ├── train.py           # Boucle d'entraînement
│   └── main_seq.py         # Script principal
│
└── requirements.txt
```

## Partie 1 — MLP (classification tabulaire)

Classification binaire sur le dataset *Breast Cancer Wisconsin* (scikit-learn). Le script compare plusieurs architectures (`MLPSequential`, `MLPClass`) et stratégies d'initialisation des poids (gaussienne, constante, Xavier), puis sélectionne le meilleur modèle selon la perte de validation.

## Partie 2 — CNN (classification d'images)

Implémentation de LeNet-5 et d'une variante améliorée (BatchNorm, ReLU, Dropout, MaxPool) entraînées sur FashionMNIST. Le module `manual_ops.py` réimplémente la corrélation croisée 2D et le pooling à la main, comparés aux opérations PyTorch natives. Une étude d'ablation (`AblationCNN`) permet de tester l'effet du padding, du stride, du type de pooling et du nombre de filtres.

## Partie 3 — Modèles séquentiels

Corpus réel **Tatoeba fra-eng** (paires de phrases anglais → français, source : manythings.org/anki,
également utilisée par le livre *Dive into Deep Learning*). Le corpus est **téléchargé
automatiquement** au premier lancement (`data_prep.py`) et mis en cache dans `partie_3_seq2seq/data/`.
En l'absence de connexion Internet, un petit corpus de secours réel mais écrit à la main
(`data/fra_sample_fallback.tsv`, ~100 phrases) prend le relais automatiquement, avec un message
d'avertissement explicite dans les logs.

- **Modèle de langage** (RNN, LSTM ou GRU au choix) : prédiction du prochain token, entraîné sur les
  phrases anglaises du corpus réel.
- **Seq2Seq** : véritable traduction anglais → français avec encodeur/décodeur GRU, teacher forcing,
  illustration de l'effet du gradient clipping (norme du gradient avant/après), puis évaluation par
  **BLEU** et comparaison **décodage glouton vs beam search** sur des phrases réelles du jeu de test.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Chaque partie s'exécute indépendamment depuis son propre dossier :

```bash
cd partie_1_mlp && python main_mlp.py
cd partie_2_cnn && python main_cnn.py
cd partie_3_seq2seq && python main_seq.py
```

## Technologies

PyTorch, torchvision, torchtext, scikit-learn, pandas, numpy, matplotlib, seaborn.
