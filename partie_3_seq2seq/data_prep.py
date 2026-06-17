
import re
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader


# Tokens spéciaux
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class SimpleTextDataset(Dataset):
    """Dataset pour la prédiction du prochain token ou la traduction."""
    def __init__(self, sequences, pad_idx):
        self.sequences = sequences
        self.pad_idx = pad_idx
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


class Vocabulary:
    """Gestion du vocabulaire et des conversions mot->idx et idx->mot."""
    def __init__(self, special_tokens):
        self.special_tokens = special_tokens
        self.token2idx = {token: i for i, token in enumerate(special_tokens)}
        self.idx2token = {i: token for i, token in enumerate(special_tokens)}
        
    def build_vocab(self, texts, min_freq=1):
        """Construit le vocabulaire à partir d'une liste de textes."""
        tokens = []
        for text in texts:
            tokens.extend(self.tokenize(text))
        counter = Counter(tokens)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                
    def tokenize(self, text):
        """Tokenisation basique (séparation sur les espaces et la ponctuation)."""
        # Séparer la ponctuation des mots
        text = re.sub(r'([.,!?;])', r' \1 ', text)
        return text.lower().strip().split()
    
    def text_to_sequence(self, text, add_eos=True, add_sos=False):
        """Convertit un texte en séquence d'idx."""
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
        """Convertit une séquence d'idx en texte."""
        tokens = []
        for idx in sequence:
            token = self.idx2token.get(idx, UNK_TOKEN)
            if token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
                if token == EOS_TOKEN:
                    break
                continue
            tokens.append(token)
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.token2idx)


def pad_sequences(sequences, pad_idx):
    """Padde des séquences à la longueur maximale du batch."""
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padded_seq = seq + [pad_idx] * (max_len - len(seq))
        padded.append(padded_seq)
    return padded


def prepare_language_modeling_data(texts, vocab, batch_size=32):
    """Prépare les données pour la prédiction du prochain token."""
    sequences = []
    for text in texts:
        seq = vocab.text_to_sequence(text, add_eos=False)
        # Créer des exemples (input, target) : input = seq[:-1], target = seq[1:]
        if len(seq) > 1:
            sequences.append(seq)
    
    # Créer le dataset et DataLoader
    dataset = SimpleTextDataset(sequences, vocab.token2idx[PAD_TOKEN])
    
    def collate_fn(batch):
        input_sequences = [seq[:-1] for seq in batch]
        target_sequences = [seq[1:] for seq in batch]
        padded_inputs = pad_sequences(input_sequences, vocab.token2idx[PAD_TOKEN])
        padded_targets = pad_sequences(target_sequences, vocab.token2idx[PAD_TOKEN])
        return torch.tensor(padded_inputs, dtype=torch.long), torch.tensor(padded_targets, dtype=torch.long)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


def get_sample_texts():
    """Retourne un petit corpus textuel pour l'entraînement."""
    return [
        "the quick brown fox jumps over the lazy dog",
        "a bird in the hand is worth two in the bush",
        "the early bird catches the worm",
        "practice makes perfect",
        "where there is a will there is a way",
        "no pain no gain",
        "the pen is mightier than the sword",
        "knowledge is power",
        "time flies when you are having fun",
        "actions speak louder than words",
        "all good things must come to an end",
        "better late than never"
    ]


if __name__ == "__main__":
    # Test du data_prep
    texts = get_sample_texts()
    vocab = Vocabulary(SPECIAL_TOKENS)
    vocab.build_vocab(texts)
    print(f"Taille du vocabulaire : {len(vocab)}")
    
    # Test de conversion
    test_text = "the quick brown fox"
    seq = vocab.text_to_sequence(test_text, add_eos=True)
    print(f"\nTexte : {test_text}")
    print(f"Séquence : {seq}")
    print(f"Retour en texte : {vocab.sequence_to_text(seq)}")
