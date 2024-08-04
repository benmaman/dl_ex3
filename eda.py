"""
this script is used to do EDA for the lyrics data and the midi data


"""

import pandas as pd
import numpy as np
import pretty_midi  
import matplotlib.pyplot as plt
import re
import seaborn as sns
import os



# %%-------------------------------lyrics data EDA--------------------------------

# load the data
lyricst = pd.read_csv('lyrics_train_set.csv')

def lyrics_eda(data):
    """
    this function is used to do EDA for the lyrics data
    :param data: the lyrics data
    :return: None
    """
    # number of  words in each lyrics
    num_words = data['lyrics']
    # remove all '&'
    num_words = num_words.replace('&', '')
    # get the number of words in each lyrics
    num_words = len(num_words.split(' '))
    data['num_words'] = num_words

    sentences = data['lyrics'].split('&')
    avg_num_words_per_sentence = []
    for sentence in sentences:
        # get the number of words in each sentence
        sn_words = sentence.strip().split(' ')
        avg_num_words_per_sentence.append(len(sn_words))
    data['avg_num_words_per_sentence'] = np.mean(avg_num_words_per_sentence)


    return data

def dict_of_freq_words(data,dict_of_w):
    """
    this function is used to get the frequency of the words in the lyrics data
    :param data: the lyrics data
    :param dict_of_w: the dictionary of the words
    :return: the dictionary of the words
    """
    lyrics = data['lyrics']
    # Lowercase and remove characters between parentheses
    cleaned_lyrics = lyrics.lower()
    cleaned_lyrics = re.sub(r'\(.*?\)', '', cleaned_lyrics)
    
    # Split into words, then add 'eos' at the end
    words = cleaned_lyrics.split()
    for word in words:
        dict_of_w[word] = dict_of_w.get(word, 0) + 1
    return dict_of_w



# run the function
lyricst = lyricst.apply(lyrics_eda, axis=1)
# get the frequency of the words
dict_of_w = {}
lyricst.apply(dict_of_freq_words, args=(dict_of_w,), axis=1)


# plot the number of words in each lyrics
plt.figure(figsize=(10, 6))
sns.histplot(lyricst['num_words'], kde=True)
plt.title('Number of words in every song')
plt.xlabel('Number of words')
plt.ylabel('Frequency')
plt.savefig('num_words.png',dpi=300)
plt.show()

# plot the average number of words in each sentence
plt.figure(figsize=(10, 6))
sns.histplot(lyricst['avg_num_words_per_sentence'], kde=True)
plt.title('Average number of words in every sentence per song')
plt.xlabel('Average number of words')
plt.ylabel('Frequency')
plt.savefig('avg_num_words_per_sentence.png',dpi=300)
plt.show()


# get the top 10 words  
dict_of_w = dict(sorted(dict_of_w.items(), key=lambda item: item[1], reverse=True))
dict_of_w = dict(list(dict_of_w.items())[:10])
# plot the top 10 words
plt.figure(figsize=(10, 6))
plt.bar(dict_of_w.keys(), dict_of_w.values())
plt.title('Top 10 words in the lyrics data')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.savefig('top_10_words.png',dpi=300)
plt.show()

# plot dist of diffrent artists
g_artist = lyricst.groupby('singer').size().reset_index(name='counts')  
plt.figure(figsize=(10, 6))
# hist plot
sns.histplot(g_artist['counts'], kde=True)
plt.title('Distribution of number of songs per artist')
plt.xlabel('Number of songs')
plt.ylabel('Frequency')
plt.savefig('artist_dist.png',dpi=300)
plt.show()

# %%-------------------------------midi data EDA--------------------------------

lengt_dict = {}
beats_dict = {}
tempo_dict = {}
tempo_changes_dict = {}
piano_roll_dict = {}
for file in os.listdir('midi_files'):
    if file.endswith('.mid'):
        try:
            print(f'Processing {file}')
            midi_data = pretty_midi.PrettyMIDI('midi_files/' + file)
        except:
            print(f'Error processing {file}')
            continue
        # get the length of the midi file
        lengt_dict[file] = midi_data.get_end_time()
        # get the beats per minute
        beats_dict[file] = midi_data.estimate_tempo()
        # get the tempo changes
        # get the tempo
        tempo_dict[file] = len(midi_data.get_tempo_changes()[1])


# plot the length of the midi files
plt.figure(figsize=(10, 6))
sns.histplot(lengt_dict.values(), kde=True)
plt.title('Length of the midi files')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.savefig('length.png',dpi=300)
plt.show()

# plot the beats per minute
plt.figure(figsize=(10, 6))
sns.histplot(beats_dict.values(), kde=True)
plt.title('Beats per minute of the midi files')
plt.xlabel('Beats per minute')
plt.ylabel('Frequency')
plt.savefig('beats.png',dpi=300)
plt.show()

# plot the tempo changes
plt.figure(figsize=(10, 6))
sns.histplot(tempo_dict.values(), kde=True,bins=25)
plt.title('number of tempo changes in the midi files')
plt.xlabel('Number of tempo changes')
plt.ylabel('Frequency')
plt.savefig('tempo_changes.png',dpi=300)
plt.show()




