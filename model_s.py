import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_length):
        self.vocab = vocab
        self.seq_length = seq_length
        self.data = self.prepare_data(texts)

    def prepare_data(self, texts):
        sequences = []
        for line in texts:
            tokenized_line = word_tokenize(line)
            for i in range(1, len(tokenized_line)):
                sequence = tokenized_line[:i + 1]
                sequences.append(sequence)
        
        max_sequence_len = max([len(seq) for seq in sequences])
        input_sequences = np.array([np.pad([self.vocab.get(word, 0) for word in seq], 
                                            (max_sequence_len - len(seq), 0), 
                                            mode='constant') 
                                    for seq in sequences])
        xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
        ys = np.zeros((len(labels), len(self.vocab) + 1), dtype=np.float32)
        for i, label in enumerate(labels):
            ys[i, label] = 1
        return xs, ys

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return torch.tensor(self.data[0][idx], dtype=torch.long), torch.tensor(self.data[1][idx], dtype=torch.float32)


class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_weights, hidden_dim, output_dim):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
        self.lstm = nn.LSTM(embedding_weights.size(1), hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    history = {'accuracy': []}
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == batch_y.argmax(dim=1)).sum().item()
        accuracy = total_correct / len(dataloader.dataset)
        history['accuracy'].append(accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}, Accuracy: {accuracy}')
    return history


def plot_history(history, metric):
    plt.plot(history[metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.show()


def generate_text(model, tokenizer, seed_text, next_words, max_sequence_len):
    model.eval()
    vocab = tokenizer.word_index
    reverse_vocab = {index: word for word, index in vocab.items()}
    for _ in range(next_words):
        token_list = [vocab.get(word, 0) for word in word_tokenize(seed_text)]
        token_list = torch.tensor([token_list], dtype=torch.long)
        token_list = torch.nn.functional.pad(token_list, (max_sequence_len - token_list.size(1), 0), "constant", 0)
        predicted = model(token_list).argmax(dim=1).item()
        output_word = reverse_vocab.get(predicted, "")
        seed_text += " " + output_word
    return seed_text


def main():
    # Load and preprocess the data
    file_path = '/tmp/irish-lyrics-eof.txt'
    data = open(file_path).read().lower().split("\n")
    
    # Tokenization and vocabulary creation
    tokenizer = word_tokenize
    tokenized_corpus = [tokenizer(sentence) for sentence in data]
    word_counts = Counter([word for sentence in tokenized_corpus for word in sentence])
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common())}
    total_words = len(vocab) + 1
    max_sequence_len = max([len(sentence) for sentence in tokenized_corpus])

    # Train Word2Vec model
    word2vec_model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    embedding_weights = np.zeros((total_words, 100))
    for word, idx in vocab.items():
        if word in word2vec_model.wv:
            embedding_weights[idx] = word2vec_model.wv[word]

    embedding_weights = torch.tensor(embedding_weights, dtype=torch.float32)

    # Create dataset and dataloader
    dataset = TextDataset(data, vocab, max_sequence_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define model
    model = TextGenerationModel(total_words, embedding_weights, 150, total_words)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    history = train_model(model, dataloader, criterion, optimizer, num_epochs=100)

    # Plot accuracy
    plot_history(history, 'accuracy')

    # Generate new text
    seed_text = "I've got a bad feeling about this"
    next_words = 100
    generated_text = generate_text(model, vocab, seed_text, next_words, max_sequence_len - 1)
    print(generated_text)


if __name__ == "__main__":
    main()



# # Example: Assume each sentence is paired with one audio clip
combined_features = []

for sequence in padded_sequences:
    # Example embedding of the sequence (dummy example)
    text_embedding = np.random.rand(maxlen, 100)  # Replace with actual text embeddings
    
    # Repeat audio features to match the sequence length (timesteps)
    audio_embedding = np.tile(mfcc, (maxlen, 1))
    
    # Combine text and audio embeddings
    combined_embedding = np.concatenate((text_embedding, audio_embedding), axis=1)
    combined_features.append(combined_embedding)

combined_features = np.array(combined_features)