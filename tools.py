import os
import re
import numpy 
from models import *
def generate_sequences(data, sequence_length, word2vec, word_to_idx):
    input_sequences = []
    target_sequences = []
    midi_sequences = []

    for _, row in data.iterrows():
        lyrics = row['lyrics']
        midi_vector = row.iloc[-50:].values.tolist()  # Get the MIDI embedding for the current song
        
        # Lowercase and remove characters between parentheses
        cleaned_lyrics = lyrics.lower()
        cleaned_lyrics = re.sub(r'\(.*?\)', '', cleaned_lyrics)
        
        # Split into words and add 'eos' at the end
        words = cleaned_lyrics.split()
        words.append('eos')

        if len(words) >= sequence_length:
            for i in range(len(words) - sequence_length):
                input_sequences.append(words[i:i + sequence_length])
                target_sequences.append(words[i + 1:i + sequence_length + 1])
                midi_sequences.append(midi_vector)

    input_vectors = [[word2vec.wv[word] for word in sequence] for sequence in input_sequences]
    target_indices = [[word_to_idx[word] for word in sequence] for sequence in target_sequences]

    return input_vectors, target_indices, midi_sequences



def low_case_name_file(midi_folder):
    """rename the name of the folder in order to import the name of the files.

    Args:
        midi_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    # List all files in the folder
    files = os.listdir(midi_folder)

    # Loop over all files in the folder
    for file in files:
        # Check if the file is a MIDI file
        if file.endswith('.mid'):
            # Get the current file path
            old_file_path = os.path.join(midi_folder, file)
            
            # Convert the file name to lowercase
            new_file_name = file.lower()
            new_file_path = os.path.join(midi_folder, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{file}' to '{new_file_name}'")

    return print("All MIDI files have been renamed to lowercase.")


def find_exact_index(df, midi_embedding):

    """
    Find the exact index in the DataFrame that matches the given MIDI embedding.

    Args:
        df (pd.DataFrame): DataFrame containing the MIDI embeddings.
        midi_embedding (list): List containing the 50-column MIDI embedding.

    Returns:
        list: List of indices that match the given MIDI embedding.
    """
    # Create a boolean mask for rows that match the MIDI embedding
    mask = (df.iloc[:, -50:] == midi_embedding).all(axis=1)
    
    # Get the indices of the matching rows
    matching_indices = df[mask].index.tolist()
    song,singer=df.iloc[matching_indices,0].values[0],df.iloc[matching_indices,1].values[0]

    return song,singer