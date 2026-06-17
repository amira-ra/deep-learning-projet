
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    """Modèle de langage (RNN/LSTM/GRU) pour prédiction du prochain token."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, rnn_type='lstm'):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type.lower()
        
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError(f"Type de RNN non supporté : {rnn_type}")
            
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch_size, seq_len)
            hidden: état caché initial (optionnel)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            hidden: nouvel état caché
        """
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        output, hidden = self.rnn(x, hidden)
        logits = self.fc(output)  # (batch_size, seq_len, vocab_size)
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialise l'état caché."""
        weight = next(self.parameters()).data
        num_layers = self.rnn.num_layers
        hidden_dim = self.rnn.hidden_size
        
        if self.rnn_type == 'lstm':
            return (
                weight.new(num_layers, batch_size, hidden_dim).zero_().to(device),
                weight.new(num_layers, batch_size, hidden_dim).zero_().to(device)
            )
        else:
            return weight.new(num_layers, batch_size, hidden_dim).zero_().to(device)


class Encoder(nn.Module):
    """Encodeur Seq2Seq (GRU/LSTM)."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, rnn_type='gru'):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type.lower()
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError(f"Type de RNN non supporté pour l'encodeur : {rnn_type}")
            
    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        return output, hidden


class Decoder(nn.Module):
    """Décodeur Seq2Seq (GRU/LSTM) avec Teacher Forcing."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, rnn_type='gru'):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type.lower()
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError(f"Type de RNN non supporté pour le décodeur : {rnn_type}")
            
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        logits = self.fc(output)
        return logits, hidden


class Seq2Seq(nn.Module):
    """Modèle Seq2Seq complet."""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: séquence source (batch_size, src_len)
            trg: séquence cible (batch_size, trg_len)
            teacher_forcing_ratio: probabilité d'utiliser Teacher Forcing
        Returns:
            outputs: logits du décodeur (batch_size, trg_len, vocab_size)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        # Initialiser les outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Passer la source dans l'encodeur
        _, hidden = self.encoder(src)
        
        # Premier input du décodeur: <SOS> token
        decoder_input = trg[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        for t in range(1, trg_len):
            # Passer l'input courant dans le décodeur
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t] = decoder_output.squeeze(1)
            
            # Décider si on utilise Teacher Forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Prochain input: vrai token (si Teacher Forcing) ou token prédit
            top1 = decoder_output.argmax(2)
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1
            
        return outputs


def greedy_decode(model, vocab, src, max_len=20, device='cpu'):
    """Décodage Glouton (Greedy Search)."""
    model.eval()
    src = src.to(device)
    
    with torch.no_grad():
        _, hidden = model.encoder(src)
        
        # Commencer avec <SOS>
        decoder_input = torch.tensor([[vocab.token2idx['<SOS>']]]).to(device)
        decoded_tokens = []
        
        for _ in range(max_len):
            decoder_output, hidden = model.decoder(decoder_input, hidden)
            top1 = decoder_output.argmax(2)
            token_idx = top1.item()
            
            if token_idx == vocab.token2idx['<EOS>']:
                break
                
            decoded_tokens.append(token_idx)
            decoder_input = top1
            
    return decoded_tokens


def beam_search_decode(model, vocab, src, k=3, max_len=20, device='cpu'):
    """Décodage par Beam Search."""
    model.eval()
    src = src.to(device)
    
    with torch.no_grad():
        _, hidden = model.encoder(src)
        
        # Initialiser le faisceau: (sequence, log_prob, hidden)
        beams = [
            ([vocab.token2idx['<SOS>']], 0.0, hidden)
        ]
        
        completed = []
        
        for _ in range(max_len):
            new_beams = []
            
            for seq, log_prob, hidden in beams:
                # Dernier token
                last_token = torch.tensor([[seq[-1]]]).to(device)
                
                decoder_output, new_hidden = model.decoder(last_token, hidden)
                
                # Top k tokens
                log_probs = F.log_softmax(decoder_output.squeeze(1), dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, k)
                
                for i in range(k):
                    token = topk_indices[0][i].item()
                    new_log_prob = log_prob + topk_log_probs[0][i].item()
                    new_seq = seq + [token]
                    
                    if token == vocab.token2idx['<EOS>']:
                        completed.append((new_seq, new_log_prob))
                    else:
                        new_beams.append((new_seq, new_log_prob, new_hidden))
                        
            # Garder seulement les k meilleurs nouveaux faisceaux
            if not new_beams:
                break
                
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:k]
            
        # Ajouter les faisceaux non complétés
        completed.extend(beams)
        
        # Retourner la meilleure séquence (sans <SOS> et <EOS>)
        best_seq = max(completed, key=lambda x: x[1])[0]
        return best_seq[1:] if best_seq and best_seq[0] == vocab.token2idx['<SOS>'] else best_seq


if __name__ == "__main__":
    # Test rapide des modèles
    print("Modèles définis avec succès!")
