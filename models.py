import pandas as pd
import re
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class LyricsDataset(Dataset):
    """
    Dataset class for lyrics data. This class takes input vectors and target indices 
    and provides them as PyTorch tensors.
    """
    def __init__(self, input_vectors, target_indices, midi_vectors):
        self.input_vectors = np.array(input_vectors)
        self.target_indices = np.array(target_indices)
        self.midi_vectors = np.array(midi_vectors)

    def __len__(self):
        return len(self.input_vectors)

    def __getitem__(self, idx):
        input_vec = torch.tensor(self.input_vectors[idx], dtype=torch.float32)
        target_idx = torch.tensor(self.target_indices[idx], dtype=torch.long)
        midi_vec = torch.tensor(self.midi_vectors[idx], dtype=torch.float32)
        return input_vec, target_idx, midi_vec

class LyricsGenerator(nn.Module):  
    """
    LSTM-based model for lyrics generation. This model uses an embedding layer followed by 
    an LSTM and a fully connected layer to predict the next word in the sequence.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,dropout):
        super(LyricsGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim , hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x,midi, hidden):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor containing word indices.
            hidden (tuple): Hidden state and cell state for the LSTM.
            
        Returns:
            out (torch.Tensor): Output predictions from the fully connected layer.
            hidden (tuple): Updated hidden state and cell state.
        """
        combined = torch.cat((x, midi.unsqueeze(1).repeat(1, x.size(1), 1)), dim=2)
        out, hidden = self.lstm(combined, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state and cell state for the LSTM.
        
        Args:
            batch_size (int): Batch size for the hidden states.
            
        Returns:
            tuple: Initialized hidden state and cell state.
        """
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())

class MergeLyricsGenerator(nn.Module):  
    """
    LSTM-based model for lyrics generation that processes text and MIDI features in parallel.
    This model first processes word indices and MIDI features separately, then concatenates
    the resulting features and feeds them into an LSTM to predict the next word in the sequence.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,dropout,midi_dim=50):
        """    

        Initializes the MergeModel with embedding layers for text and MIDI data,
        an LSTM layer for sequence modeling, and a fully connected layer for output.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique words).
            embedding_dim (int): Dimension of the word and MIDI embeddings.
            midi_dim (int): Input dimension of the MIDI features.
            hidden_dim (int): Number of features in the hidden state of the LSTM.
            num_layers (int): Number of recurrent layers in the LSTM.
        """
        super(MergeLyricsGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Embedding for words
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Dense layer for MIDI processing
        self.midi_dense = nn.Linear(midi_dim, 5)
        self.relu = nn.ReLU()


    
        # LSTM for combined features
        self.lstm = nn.LSTM(embedding_dim+5, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, word_embedded, midi_features, hidden):
        """
        Forward pass through the model.

        Args:
            word_embedded (torch.Tensor): Tensor containing word emdbeeded using word2vec (batch_size, seq_length).
            midi_features (torch.Tensor): Tensor containing MIDI features (batch_size, midi_dim).
            hidden (tuple): Tuple containing the initial hidden state and cell state for the LSTM.

        Returns:
            out (torch.Tensor): Tensor containing the output predictions (batch_size, seq_length, vocab_size).
            hidden (tuple): Updated hidden state and cell state.
        """
        
        # Process MIDI features
        midi_processed = self.midi_dense(midi_features)  # Shape: (batch_size, embedding_dim)
        midi_processed = self.relu(midi_processed)
        midi_processed = midi_processed.unsqueeze(1).repeat(1, word_embedded.size(1), 1)
        
        # Combine word and MIDI features
        combined = torch.cat((word_embedded, midi_processed), dim=2)  # Shape: (batch_size, seq_length, 2 * embedding_dim)
        
        # LSTM
        out, hidden = self.lstm(combined, hidden)
        
        # Fully connected layer to vocab size
        out = self.fc(out)
        
        return out, hidden



    def init_hidden(self, batch_size):
        """
        Initializes the hidden state and cell state for the LSTM.

        Args:
            batch_size (int): The batch size used in training.

        Returns:
            tuple: Initialized hidden state and cell state tensors, each of shape (num_layers, batch_size, hidden_dim).
        """
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())




def generate_text(seed_text, model,sequence_length, max_length, vocab_size, word_to_idx, idx_to_word, word2vec, midi_embedding):
    """
    Generates text using the trained LSTM model starting from a seed text.
    
    Args:
        seed_text (str): Initial text to start the generation.
        model (LyricsGenerator): Trained lyrics generation model.
        max_length (int): Maximum length of the generated text.
        vocab_size (int): Size of the vocabulary.
        word_to_idx (dict): Dictionary mapping words to their indices.
        idx_to_word (dict): Dictionary mapping indices to their words.
        word2vec (gensim.models.Word2Vec): Pre-trained Word2Vec model.
        midi_embedding (numpy.ndarray): MIDI embedding to use during generation.
        
    Returns:
        str: Generated text.
    """
    model.eval()
    words = seed_text.lower().split()
    input_indices = [word2vec.wv[word] for word in words]
    x = torch.tensor([input_indices], dtype=torch.float32)  # Add batch dimension

    midi = torch.tensor(midi_embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    state_h, state_c = model.init_hidden(1)  # Initialize hidden state with batch size 1

    generated_words = words.copy()
    for _ in range(max_length):
        y_pred, (state_h, state_c) = model(x, midi, (state_h, state_c))
        last_word_logits = y_pred[0, -1]  # Get the last time step
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_idx = np.random.choice(len(last_word_logits), p=p)
        new_word = idx_to_word[word_idx]
        generated_words.append(new_word)

        # Update x for the next prediction
        input_text = generated_words[-sequence_length:]
        input_indices = [word2vec.wv[word] for word in input_text]
        x = torch.tensor([input_indices], dtype=torch.float32)  # Add batch dimension

    # Edit the final lyrics - take the first eos or last &
    final_lyrics = []
    if 'eos' in generated_words:
        # Find the index of 'eos' and take all words before it
        index = generated_words.index('eos')
        final_lyrics.append(generated_words[:index])
    if '&' in generated_words:
        # If there is no 'eos', take all the words
        last_index = len(generated_words) - 1 - generated_words[::-1].index('&')
        final_lyrics.append(generated_words[:last_index])
    else:
        final_lyrics.append(generated_words)

    return ' '.join(generated_words)