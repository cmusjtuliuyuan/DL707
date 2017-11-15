import codecs
import os
import numpy as np
import matplotlib.pyplot as plt

def load_sentences(path):
    """
    Load datasets
    """
    sentences = []
    for line in codecs.open(path, 'r', 'utf8'):
        sentences.append([u'START',]+line.lower().split()+[u'END',])
    return sentences

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico, vocabulary_size=8000):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), 
            key=lambda x: (-x[1], x[0]))[:vocabulary_size-1] # -1 caused by UNK
    sorted_items.append((u'UNK',0))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def sentences2index(sentences, dictionary):
    dataset = []
    word_to_id = dictionary['word_to_id']
    for sentence in sentences:
        for i in range(3,len(sentence)):
            grams_4 = [word_to_id[sentence[index]] if sentence[index] in word_to_id else word_to_id[u'UNK']
                      for index in range(i-3, i+1)]
            dataset.append(grams_4)
    dataset = np.array(dataset)
    return dataset

def get_train_dataset(path):
    sentences = load_sentences(path)
    dico = create_dico(sentences)
    word_to_id, id_to_word = create_mapping(dico)
    dictionary={'word_to_id': word_to_id, 'id_to_word':id_to_word}
    dataset = sentences2index(sentences, dictionary)
    return dataset, dictionary

def get_val_dataset(path, dictionary):
    sentences = load_sentences(path)
    dataset = sentences2index(sentences, dictionary)
    return dataset

def plot_dataset(dataset, dictionary):
    dico = {}
    for grams_4 in dataset:
        item = tuple(grams_4)
        if item not in dico:
            dico[item] = 1
        else:
            dico[item] += 1
    sorted_items = sorted(dico.items(), 
            key=lambda x: (-x[1], x[0]))
    

    print 'top 50 4-grams are:'
    id_to_word = dictionary['id_to_word']
    for id_tuple, freq in sorted_items[:50]:
        grams_4 = list(id_tuple)
        print id_to_word[grams_4[0]],id_to_word[grams_4[1]],\
              id_to_word[grams_4[2]],id_to_word[grams_4[3]], freq
    # Plot the histogram of 4 grams
    frequency = sorted(dico.values(), reverse=True)
    fig, ax = plt.subplots()
    plt.yscale('log', nonposy='clip')
    n, bins, histpatches = ax.hist(frequency, 20, facecolor='blue')
    plt.xlabel('Appear times')
    plt.ylabel('Number of 4-grams')
    plt.title('Histogram of 4-grams')
    plt.grid(True)
    plt.savefig('higtogram4grams.png')


#traindataset, dictionary = get_train_dataset('train.txt')
#valdataset = get_val_dataset('val.txt', dictionary)
#plot_dataset(traindataset, dictionary)