"""
Préparation des données pour la Partie III (modèles séquentiels / Seq2Seq).

Le corpus utilisé est un corpus réel de traduction anglais -> français issu du
projet Tatoeba (distribué via manythings.org/anki, source également utilisée
par de nombreux cours de deep learning, dont le livre "Dive into Deep
Learning"). Le fichier est téléchargé automatiquement au premier lancement et
mis en cache localement dans le dossier data/.

Si aucune connexion Internet n'est disponible (par exemple lors d'une
correction hors-ligne), un petit corpus de secours réel mais écrit à la main
(data/fra_sample_fallback.tsv, ~100 phrases) est utilisé à la place, afin que
le script reste exécutable dans tous les cas. Ce repli est clairement signalé
dans les logs et NE remplace PAS le corpus réel utilisé pour les résultats
présentés dans le rapport.
"""

import os
import re
import zipfile
import urllib.request
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------
# Tokens spéciaux
# ----------------------------------------------------------------------
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ZIP_PATH = os.path.join(DATA_DIR, "fra-eng.zip")
RAW_TXT_PATH = os.path.join(DATA_DIR, "fra.txt")
FALLBACK_PATH = os.path.join(DATA_DIR, "fra_sample_fallback.tsv")

# Sources possibles pour le corpus réel (on essaie dans l'ordre)
DOWNLOAD_URLS = [
    "http://www.manythings.org/anki/fra-eng.zip",
    "http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip",
]


# ----------------------------------------------------------------------
# Téléchargement et lecture du corpus réel
# ----------------------------------------------------------------------
def _download_fra_eng(timeout=10):
    """Tente de télécharger le corpus réel Tatoeba fra-eng. Retourne True si succès."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_TXT_PATH):
        return True

    for url in DOWNLOAD_URLS:
        try:
            print(f"Tentative de téléchargement du corpus réel depuis : {url}")
            urllib.request.urlretrieve(url, ZIP_PATH)
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                # Le fichier dans l'archive s'appelle 'fra.txt'
                names = [n for n in z.namelist() if n.endswith("fra.txt")]
                if not names:
                    continue
                with z.open(names[0]) as src, open(RAW_TXT_PATH, "wb") as dst:
                    dst.write(src.read())
            print("Téléchargement réussi : corpus réel Tatoeba fra-eng disponible.")
            return True
        except Exception as e:
            print(f"  Échec du téléchargement depuis {url} : {e}")
            continue
    return False


def read_fra_eng_pairs(num_examples=None):
    """
    Retourne une liste de paires (phrase_en, phrase_fr) issues du corpus réel.

    Essaie d'abord de télécharger le corpus Tatoeba complet (plusieurs
    dizaines de milliers de phrases). En cas d'échec (pas de connexion),
    bascule sur le corpus de secours local (data/fra_sample_fallback.tsv).
    """
    ok = _download_fra_eng()

    if ok and os.path.exists(RAW_TXT_PATH):
        source = RAW_TXT_PATH
        print(f"Utilisation du corpus réel téléchargé : {source}")
    else:
        source = FALLBACK_PATH
        print(
            "ATTENTION : impossible de télécharger le corpus Tatoeba "
            f"(pas de connexion ?). Utilisation du corpus de secours local : {source}"
        )

    pairs = []
    with open(source, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            en, fr = parts[0], parts[1]
            pairs.append((en, fr))

    if num_examples is not None:
        pairs = pairs[:num_examples]

    print(f"Nombre de paires de phrases chargées : {len(pairs)}")
    return pairs


# ----------------------------------------------------------------------
# Nettoyage et tokenisation
# ----------------------------------------------------------------------
def preprocess_text(text):
    """Nettoyage simple : espaces insécables, minuscules, espacement de la ponctuation."""
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    text = text.lower().strip()

    def needs_space(prev_char, char):
        return char in ',.!?' and prev_char != ' '

    out = []
    for i, ch in enumerate(text):
        if i > 0 and needs_space(text[i - 1], ch):
            out.append(' ')
        out.append(ch)
    text = ''.join(out)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def basic_tokenize(text):
    """Tokenisation par espaces après nettoyage."""
    return preprocess_text(text).split(' ')


# ----------------------------------------------------------------------
# Vocabulaire
# ----------------------------------------------------------------------
class Vocabulary:
    """Gestion du vocabulaire et des conversions mot <-> idx pour une langue donnée."""

    def __init__(self, special_tokens):
        self.special_tokens = special_tokens
        self.token2idx = {token: i for i, token in enumerate(special_tokens)}
        self.idx2token = {i: token for i, token in enumerate(special_tokens)}

    def build_vocab(self, texts, min_freq=1):
        """Construit le vocabulaire à partir d'une liste de phrases brutes (non tokenisées)."""
        token_lists = [basic_tokenize(t) for t in texts]
        self.build_vocab_from_tokens(token_lists, min_freq=min_freq)

    def build_vocab_from_tokens(self, token_lists, min_freq=1):
        """Construit le vocabulaire à partir de listes de tokens déjà tokenisées."""
        counter = Counter(tok for tokens in token_lists for tok in tokens)
        for token, freq in sorted(counter.items(), key=lambda x: -x[1]):
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def tokenize(self, text):
        return basic_tokenize(text)

    def text_to_sequence(self, text, add_eos=True, add_sos=False):
        tokens = self.tokenize(text)
        sequence = []
        if add_sos:
            sequence.append(self.token2idx[SOS_TOKEN])
        for token in tokens:
            sequence.append(self.token2idx.get(token, self.token2idx[UNK_TOKEN]))
        if add_eos:
            sequence.append(self.token2idx[EOS_TOKEN])
        return sequence

    def sequence_to_text(self, sequence):
        tokens = []
        for idx in sequence:
            token = self.idx2token.get(idx, UNK_TOKEN)
            if token in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                if token == EOS_TOKEN:
                    break
                continue
            tokens.append(token)
        return ' '.join(tokens)

    def __len__(self):
        return len(self.token2idx)


# ----------------------------------------------------------------------
# Padding / troncature (cf. fiche de synthèse RNN/Seq2Seq, section 8.3)
# ----------------------------------------------------------------------
def truncate_pad(line, num_steps, padding_token):
    """Tronque ou complète une séquence de tokens à une longueur fixe num_steps."""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def pad_sequences(sequences, pad_idx):
    """Padde des séquences à la longueur maximale du batch (longueur dynamique)."""
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]


# ----------------------------------------------------------------------
# Données pour le modèle de langage (prédiction du prochain token)
# ----------------------------------------------------------------------
class SimpleTextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def prepare_language_modeling_data(texts, vocab, batch_size=32):
    """Prépare les données pour la prédiction du prochain token à partir de phrases réelles."""
    sequences = []
    for text in texts:
        seq = vocab.text_to_sequence(text, add_eos=False)
        if len(seq) > 1:
            sequences.append(seq)

    dataset = SimpleTextDataset(sequences)

    def collate_fn(batch):
        input_sequences = [seq[:-1] for seq in batch]
        target_sequences = [seq[1:] for seq in batch]
        padded_inputs = pad_sequences(input_sequences, vocab.token2idx[PAD_TOKEN])
        padded_targets = pad_sequences(target_sequences, vocab.token2idx[PAD_TOKEN])
        return (torch.tensor(padded_inputs, dtype=torch.long),
                torch.tensor(padded_targets, dtype=torch.long))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# ----------------------------------------------------------------------
# Données pour la traduction Seq2Seq (paires source/cible réelles)
# ----------------------------------------------------------------------
class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return self.src_sequences[idx], self.trg_sequences[idx]


def prepare_translation_data(pairs, src_vocab, trg_vocab, batch_size=32):
    """
    Prépare un DataLoader de traduction à partir de vraies paires (en, fr).
    Le décodeur reçoit toujours <SOS> ... <EOS> ; l'encodeur reçoit la phrase
    source terminée par <EOS> (sans <SOS>, conformément à l'architecture
    encodeur-décodeur classique).
    """
    src_sequences, trg_sequences = [], []
    for en, fr in pairs:
        src_seq = src_vocab.text_to_sequence(en, add_eos=True, add_sos=False)
        trg_seq = trg_vocab.text_to_sequence(fr, add_eos=True, add_sos=True)
        src_sequences.append(src_seq)
        trg_sequences.append(trg_seq)

    dataset = TranslationDataset(src_sequences, trg_sequences)

    def collate_fn(batch):
        src_batch = [item[0] for item in batch]
        trg_batch = [item[1] for item in batch]
        src_padded = pad_sequences(src_batch, src_vocab.token2idx[PAD_TOKEN])
        trg_padded = pad_sequences(trg_batch, trg_vocab.token2idx[PAD_TOKEN])
        return (torch.tensor(src_padded, dtype=torch.long),
                torch.tensor(trg_padded, dtype=torch.long))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# ----------------------------------------------------------------------
# Filtrage par longueur (pratique standard pour accélérer l'entraînement
# pédagogique sur un sous-ensemble du corpus réel)
# ----------------------------------------------------------------------
def filter_pairs_by_length(pairs, max_len=10):
    filtered = []
    for en, fr in pairs:
        if len(basic_tokenize(en)) <= max_len and len(basic_tokenize(fr)) <= max_len:
            filtered.append((en, fr))
    return filtered


if __name__ == "__main__":
    # Test rapide du chargement du corpus réel
    pairs = read_fra_eng_pairs(num_examples=20000)
    pairs = filter_pairs_by_length(pairs, max_len=10)
    print(f"Paires après filtrage par longueur (<=10 tokens) : {len(pairs)}")
    print("Exemples :")
    for en, fr in pairs[:5]:
        print(f"  EN: {en}  ->  FR: {fr}")

    src_vocab = Vocabulary(SPECIAL_TOKENS)
    trg_vocab = Vocabulary(SPECIAL_TOKENS)
    src_vocab.build_vocab([p[0] for p in pairs], min_freq=2)
    trg_vocab.build_vocab([p[1] for p in pairs], min_freq=2)
    print(f"\nTaille vocabulaire source (EN) : {len(src_vocab)}")
    print(f"Taille vocabulaire cible (FR)  : {len(trg_vocab)}")
