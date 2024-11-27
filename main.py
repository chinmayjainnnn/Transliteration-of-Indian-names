# -*- coding: utf-8 -*-
"""


The goal of this project is to model Indian first names using character-level language models.


```
SAPname_SRno_assigment2
│   SAPname_SRno_assignment2.py  
│
└───SAPname_SRno
│   │
│   └───fnn
│   │   │   model.pt
│   │   │   vocab.pt
│   │   │   loss.json
│   │
│   └───rnn
│       │   model.pt
│       │   vocab.pt
│       │   loss.json

```

we have implemented different types of language models for modeling Indian names. 
There are clealry patterns in Indian names that models could learn, and we start modeling those using n-gram models, then move to neural n-gram and RNN models.



# Read and Preprocess Data
"""

# importing necessary libraries
import pandas as pd
import numpy as np
import math

import random
from collections import Counter, defaultdict

from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

"""Please note that we may change the contents of the following four files when we rerun your code, so please make sure that your solution is not specifically engineered to just these names."""

# Download the training and validation datasets
# !wget -O train_data.csv "https://docs.google.com/spreadsheets/d/1AUzwOQQbAehg_eoAMCcWfwSGhKwSAtnIzapt2wbv0Zs/gviz/tq?tqx=out:csv&sheet=train_data.csv"
# !wget -O valid_data.csv "https://docs.google.com/spreadsheets/d/1UtQErvMS-vcQEwjZIjLFnDXlRZPxgO1CU3PF-JYQKvA/gviz/tq?tqx=out:csv&sheet=valid_data.csv"

# # Download the text for evaluation
# !wget -O eval_prefixes.txt "https://drive.google.com/uc?export=download&id=1tuRLJXLd2VcDaWENr8JTZMcjFlwyRo60"
# !wget -O eval_sequences.txt "https://drive.google.com/uc?export=download&id=1kjPAR04UTKmdtV-FJ9SmDlotkt-IKM3b"

## Please do not change anything in this code block.

def read_dataframe(ds_type):
    """
    Args:
        ds_type [str] :  dataset type (train or valid)

    Returns:
        df [pandas dataframe]
    """

    df = pd.read_csv(f"{ds_type}_data.csv", header=0, index_col=0)
    df = df[~df['Name'].isna()]
    df['Name'] = df['Name'].astype(str)
    return df

# Load the training and validation datasets
train_data = read_dataframe("train")
validation_data = read_dataframe("valid")

# Read files containing prefixes and character sequences for evaluation
with open('eval_prefixes.txt', 'r') as file:
    eval_prefixes = []
    for line in file:
        eval_prefixes.append(line.strip().split(" "))

with open('eval_sequences.txt', 'r') as file:
    eval_sequences = []
    for line in file:
        eval_sequences.append(line.strip().split(" "))

print(f"Length of training data: {len(train_data)}\nLength of validation data: {len(validation_data)}")

## Please do not change anything in this code block.

START = "<s>"   # Start-of-name token
END = "</s>"    # End-of-name token
UNK = "<unk>"   # token representing out of unknown (or out of vocabulary) tokens
vocab_from_ascii = True

def build_vocab(names):
    """
    Builds a vocabulary given a list of names

    Args:
        names [list[str]]: list of names

    Returns:
        vocab [torchtext.vocab]: vocabulary based on the names

    """

    if vocab_from_ascii:
        char_counts = {chr(i):i for i in range(128)}
    else:
        char_counts = Counter("".join(names))
    # for char, count in char_counts.items():
    #     print(f"Character: {char}, Count: {count}")
    vocab = build_vocab_from_iterator(
                    char_counts,
                    specials=[UNK, START, END], #adding special tokens to the vocabulary
                    min_freq=1
                )
    vocab.set_default_index(vocab[UNK])
    # for token, index in vocab.get_stoi().items():
    #     print(f"Token: {token}, Index: {index}")
    return vocab


def tokenize_name(name):
    """
    Tokenise the name i.e. break a name into list of characters

    Args:
        name [str]: name to be tokenized

    Returns:
        list of characters
    """

    return list(str(name))


def process_data_for_input(data_iter, vocab):
    """
    Processes data for input: Breaks names into characters,
    converts out of vocabulary tokens to UNK and
    appends END token at the end of every name

    Args:
        data_iter: data iterator consisting of names
        vocab: vocabulary

    Returns:
        data_iter [list[list[str]]]: list of names, where each name is a
                                list of characters and is appended with
                                START and END tokens

    """

    vocab_set = set(vocab.get_itos())
    # convert Out Of Vocabulary (OOV) tokens to UNK tokens
    data_iter = [[char if char in vocab_set else UNK
                        for char in tokenize_name(name)] for name in data_iter]
    data_iter = [[START] + name + [END] for name in data_iter]
    # print(data_iter)
    return data_iter


def get_tokenised_text_and_vocab(ds_type, vocab=None):
    """
    Reads input data, tokenizes it, builds vocabulary (if unspecified)
    and outputs tokenised list of names (which in turn is a list of characters)

    Args:
        ds_type [str]: Type of the dataset (e.g., train, validation, test)
        vocab [torchtext.vocab]: vocabulary;
                                 If vocab is None, the function will
                                 build the vocabulary from input text.
                                 If vocab is provided, it will tokenize name
                                 according to the vocab, replacing any tokens
                                 not part of the vocab with UNK token.

    Returns:
        data_iter: data iterator for tokenized names
        vocab: vocabulary

    """

    # read the 'Name' column of the dataframe
    if ds_type=='train':
        data_iter = train_data['Name']
    elif ds_type=='valid':
        data_iter = validation_data['Name']
    else:
        data_iter = test_data['Name']

    # build vocab from input data, if vocab is unspecified
    if vocab is None:
        vocab = build_vocab(data_iter)

    # convert OOV chars to UNK, append START and END token to each name
    data_iter = process_data_for_input(data_iter, vocab)

    return data_iter, vocab

"""Let's look at some examples from the training set"""

# Look at some random examples from the training set
examples = ", ".join(random.sample(list(train_data['Name']), 5))
f"Examples from the training set: {examples}"

"""# Module 1: N-gram Language Modelling

Load and preprocess the data for n-gram models
"""

"""choose your hyperparameter and see the difference in performance"""

# CHANGE THE None VALUES TO YOUR DESIRED VALUES

# ADD YOUR CODE HERE

MAX_NAME_LENGTH = 30 # maximum length of names for generation

# Get data iterator and build vocabulary from input text

train_text, vocab = get_tokenised_text_and_vocab(ds_type='train')
validation_text, _ = get_tokenised_text_and_vocab(ds_type='valid', vocab=vocab)

# Check the size of vocabulary
# print(vocab)
vocab_size = len(vocab.get_stoi())
# print(vocab_size)

"""
Now it's time to implement an [n-gram language model](https://web.stanford.edu/~jurafsky/slp3/3.pdf).

One edge case you will need to handle is that you don't have $n-1$ prior characters at the beginning of the text.  One way to do this is by appending the `START` token $n-1$ times at the start of the name, when implementing an $n$-gram model. You may choose whichever method you like to handle this case as long as you produce a valid probability distribution (one that sums to one).

**Generating names**

To generate from a language model, we can sample one char at a time conditioning on the chars we have generated so far.

In fact there are many strategies to get better-sounding samples, such as only sampling from the top-k chars or sharpening the distribution with a temperature.  You can read more about sampling from a language model in [this](https://arxiv.org/pdf/1904.09751.pdf) paper.


We will now implement N-gram models with $N=1$ (unigram), $N=2$ (bigram), and $N=3$ (trigram).

**Utility Functions**

Implement the utility functions `get_unigram_counts`, `get_bigram_counts` and `get_trigram_counts`. You can use these functions while implementing n-gram models."""

def get_unigram_counts(corpus):
    """
    Given a corpus, calculates the unigram counts for each character in the corpus

    Args:
        corpus [list[list[str]]]: list of tokenized characters. Text is appended with END token.

    Returns:
        unigram_counts [dict [key: char, value: count]]:
            dictionary of unigram counts for each character in the corpus
        Example:
        > unigram_counts["c1"] = 5
    """
    unigram_counts={}

    for name in corpus:
        for chara in name:
            unigram_counts[chara]=unigram_counts.get(chara,0)+1
    return unigram_counts

def get_bigram_counts(corpus):
    """
    Given a corpus, calculates the bigram counts for each bigram in the corpus.
    The corpus *only* contains END tokens at the end of names.
    You may want to handle the case whhere beginning of the name
    does not have n-1 prior chars.

    Args:
        corpus [list[list[str]]]: list of tokenized text. Text is appended with END token.

    Returns:
        bigram_counts [dict[dict]]:
            nested dictionary of bigram counts for each bigram in the corpus
        Example:
        > bigram_counts["c1"]["c2"] = 5
        here bigram_counts["c1"]["c2"] represents P("c2"|"c1")
        P["c1"]["c2"] means P[char_i = "c2"|char_{i-1} = "c1"]
    """

    bigram_counts={}

    for name in corpus:
        for i in range(1,len(name)):
            if name[i-1] not in bigram_counts:
                bigram_counts[name[i-1]]={}
            bigram_counts[name[i-1]][name[i]]=bigram_counts[name[i-1]].get(name[i],0)+1
    return bigram_counts

# corpus=[]
# c1=["chinmay","shreeman","sahil"]

# for word in c1:
#     b=[]
#     for char in word:
#         b.append(char)
#     corpus.append(b)
# get_bigram_counts(corpus)

# corpus=[]
# c1=["abcdbce","bcdbcef","dbc"]

# for word in c1:
#     b=[]
#     for char in word:
#         b.append(char)
#     corpus.append(b)
# get_trigram_counts(corpus)

def get_trigram_counts(corpus):
    """
    Given a corpus, calculates the trigram counts for each trigram in the corpus.
    The corpus *only* contains END tokens at the end of names.
    You may want to handle the case where beginning of the text
    does not have n-1 prior chars.

    Args:
        corpus [list[list[token]]]: list of tokenized text. Text is appended with END token.

    Returns:
        trigram_counts [dict[dict[dict]]]:
            nested dictionary for each trigram in the corpus
        Example:
        > trigram_counts["c1"]["c2"]["c3"] = 5
        P["c1"]["c2"]["c3] means P[char_i = "c3"|char_{i-2} = "c1", char_{i-1} = "c2"]

    """

    trigram_counts={}

    for name in corpus:
        name = [START] + name
        for i in range(1,len(name)):
            if name[i-2] not in trigram_counts:
                trigram_counts[name[i-2]]={}
            if name[i-1] not in trigram_counts[name[i-2]]:
                trigram_counts[name[i-2]][name[i-1]]={}

            trigram_counts[name[i-2]][name[i-1]][name[i]] = trigram_counts[name[i-2]][name[i-1]].get(name[i],0)+1

    return trigram_counts

"""
Implementation of the n-gram language models.
All other n-gram models (unigram, bigram, etc.) would follow the same skeleton.
"""

class NGramLanguageModel(object):
    def __init__(self, train_text):
        """
        Initialise and train the model with train_text.

        Args:
            train_text [list of list]: list of tokenised names

        Returns:
            -
        """
        return

    def get_next_char_probabilities(self):
        """
        Returns a probability distribution over all chars in the vocabulary.
        Probability distribution should sum to one.

        Returns:
            P: dictionary or nested dictionary; Output format depends on n-gram
            Examples:
                for N=1 (unigram); dict[key:unigram,value:probability of unigram]
                    > P["c1"] = 0.0001
                for N=2 (bigram); dict[key:bigram_char1, value:dict[key:bigram_char2,value:probability of bigram]]
                    > P["c1"]["c2"] = 0.0001
                    P["c1"]["c2"] means P["c2"|"c1"]
                for N=3 (trigram); dict[dict[dict]]
                    > P["c1"]["c2"]["c3"] = 0.0001
                    P["c1"]["c2"]["c3] means P[char_i = "c3"|char_{i-2} = "c1", char_{i-1} = "c2"]
        """

        return


    def get_name_log_probability(self, name):
        """
        Calculates the log probability of name according to the language model

        Args:
            name [list]: list of tokens

        Returns:
            log_prob [float]: Log probability of the given name
        """
        return 0.0


    def get_perplexity(self, text):
        """
        Returns the perplexity of the model on a text as a float.

        Args:
            text [list]: a list of string tokens

        Returns:
            perplexity [float]: perplexity of the given text
        """
        return 0.0


    def generate_names(self, k, n=MAX_NAME_LENGTH, prefix=None):
        """
        Given a prefix, generate k names according to the model.
        The default prefix is None.
        You may stop the generation when n tokens have been generated,
        or when you encounter the END token.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated

        Returns:
            names [list[str]]: list of generated names
        """
        return []

    def get_most_likely_chars(self, sequence, k):
        """
        Given a sequence of characters, outputs k most likely characters after the sequence.

        Args:
            sequence [list[str]]: list of characters
            k [int]: number of most likely characters to return

        Returns:
            chars [list[str]]: *Ordered* list of most likely characters
                        (with charcater at index 0 being the most likely and
                        character at index k-1 being the least likely)

        """
        return []

## Please do not change anything in this code block.

def check_validity(model, ngram, is_neural):
    """
    Checks if get_next_char_probabilities returns a valid probability distribution
    """

    if ngram==1 or is_neural:
        P = model.get_next_char_probabilities()
        is_valid = validate_probability_distribution(P.values())
        if not is_valid:
            return is_valid

    elif ngram==2:
        P = model.get_next_char_probabilities()
        for char1 in P.keys():
            is_valid = validate_probability_distribution(list(P[char1].values()))
            if not is_valid:
                return is_valid

    elif ngram==3:
        P = model.get_next_char_probabilities()
        for char1 in P.keys():
            for char2 in P[char1].keys():
                is_valid = validate_probability_distribution(list(P[char1][char2].values()))
                if not is_valid:
                    return is_valid
    else:
        print("Enter a valid number for ngram")

    return True


def validate_probability_distribution(probs):
    """
    Checks if probs is a valid probability distribution
    """
    if not min(probs) >= 0:
        print("Negative value in probabilities")
        return False
    elif not max(probs) <= 1 + 1e-8:
        print("Value larger than 1 in probabilities")
        return False
    elif not abs(sum(probs)-1) < 1e-4:
        print("probabilities do not sum to 1")
        return False
    return True


def eval_ngram_model(model, ngram, ds, ds_name, eval_prefixes, eval_sequences, num_names=5, is_neural=False):
    """
    Runs the following evaluations on n-gram models:
    (1) checks if probability distribution returned by model.get_next_char_probabilities() sums to one
    (2) checks the perplexity of the model
    (3) generates names using model.generate_names()
    (4) generates names given a prefix using model.generate_names()
    (4) output most likely characters after a given sequence of chars using model.get_most_likely_chars()
    """

    # (1) checks if probability distributions sum to one
    is_valid = check_validity(model=model, ngram=ngram, is_neural=is_neural)
    print(f'EVALUATION probability distribution is valid: {is_valid}')

    # (2) evaluate the perplexity of the model on the dataset
    print(f'EVALUATION of {ngram}-gram on {ds_name} perplexity:',
        model.get_perplexity(ds))

    # (3) generate a few names
    generated_names = ", ".join(model.generate_names(k=num_names))
    print(f'EVALUATION {ngram}-gram generated names are {generated_names}')

    # (4) generate a few names given a prefix
    for prefix in eval_prefixes:
        generated_names_with_prefix = ", ".join(model.generate_names(k=num_names, prefix=prefix))
        prefix = ''.join(prefix)
        print(f'EVALUATION {ngram}-gram generated names with prefix {prefix} are {generated_names_with_prefix}')

    # (5) get most likely characters after a sequence
    for sequence in eval_sequences:
        most_likely_chars = ", ".join(model.get_most_likely_chars(sequence=sequence, k=num_names))
        sequence = "".join(sequence)
        print(f"EVALUATION {ngram}-gram top most likely chars after {sequence} are {most_likely_chars}")

"""## 1.1 Unigram"""

"""
Implementaion of a Unigram Model without smoothing
"""

class UnigramModel(NGramLanguageModel):
    def __init__(self, train_text):
        """
        Initialise and train the model with train_text.

        Args:
            train_text [list of list]: list of tokenised names
        """
        self.unigram_count=get_unigram_counts(train_text)
        self.unigram_total=sum(self.unigram_count.values())
        # ADD YOUR CODE HERE


    def get_next_char_probabilities(self):
        """
        Return a dictionary of probabilities for each char in the vocabulary

        Returns:
            key: char, value: probability
        """
        next_char_probabilities={}
        for char,value in self.unigram_count.items():
            next_char_probabilities[char]=self.unigram_count[char]/self.unigram_total

        # ADD YOUR CODE HERE

        return next_char_probabilities

    def get_name_log_probability(self, name):
        """
        Calculates the log probability of name according to the n-gram model

        Args:
            name [list]: list of tokens

        Returns:
            Log probability of the name [float]
        """
        name_log_probability=0
        for char in name:
            name_log_probability += np.log(self.unigram_count.get(char,0)/self.unigram_total)
        return name_log_probability



    def get_perplexity(self, text):
        """
        Returns the perplexity of the model on a text as a float.

        Args:
            text [list]: a list of string tokens

        Returns:
            perplexity of the given text [float]
        """

        # ADD YOUR CODE HERE
        # Get log probability of the text
        log_prob = sum(self.get_name_log_probability(name)/len(name) for name in text)
        # Calculate perplexity
        perplexity = math.exp(-log_prob / len(text))

        return perplexity


    def generate_names(self, k, n=MAX_NAME_LENGTH, prefix=None):
        """
        Given a prefix, generate k names according to the model.
        The default prefix is None.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated
        Returns:
            list of generated names [list]
        """
        names = []
        for _ in range(k):
            temp = []
            if prefix:
                temp.extend(prefix)
            while(len(temp)<n  ):
                next_char_prob=self.get_next_char_probabilities()
                cha=( random.choices( list(next_char_prob.keys()) ,list(next_char_prob.values()) ))[0]
                temp.append(cha)
            name_str = "".join(temp)
            names.append(name_str)
        # ADD YOUR CODE HERE
        return names


    def get_most_likely_chars(self, sequence, k):
        """
        Given a sequence of characters, outputs k most likely characters after the sequence.

        Args:
            sequence [list[str]]: list of characters
            k [int]: number of most likely characters to return

        Returns:
            chars [list[str]]: *Ordered* list of most likely characters
                        (with charcater at index 0 being the most likely and
                        character at index k-1 being the least likely)

        """
        # ADD YOUR CODE HERE
        next_char_probabilities = self.get_next_char_probabilities()
        most_likely_chars = sorted(next_char_probabilities, key=next_char_probabilities.get, reverse=True)[:k]
        return most_likely_chars

"""### Eval

**Note**: For models without smoothing, you may observe perplexity as `inf` if the validation or test set contains characters not seen in the train set
However, this should not happen for models where you implement smoothing.
"""

## Please do not change anything in this code block.

unigram_model = UnigramModel(train_text)

# Check the perplexity of the unigram model on the train set
print('unigram train perplexity:',
      unigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=unigram_model, ngram=1, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

"""### Smoothing

Implement a smoothed version of the unigram model. You may extend the `UnigramModel` class and re-use some of the functions.  For unigram model, you should implement Add-1 smoothing.

You may refer to the lecture slides or [3.5 Smoothing](https://web.stanford.edu/~jurafsky/slp3/3.pdf) for details on different smoothing technqiues.

"""

"""
Implementation of unigram model with Add-1 smoothing.

"""
class SmoothedUnigramModel(UnigramModel):

    def __init__(self, train_text):
        super().__init__(train_text)



    # You should override ONLY those functions
    # which calculate probability of a unigram.
    # You can override get_next_char_probabilities
    # or any other helper functions you use in UnigramModel
    # to calculate unigram probabilities.

    # Implement Laplace or Add-1 smoothing for the unigram model

    # ADD YOUR CODE HERE
    # self.unigram_count=get_unigram_counts(train_text)
    # self.unigram_total=sum(self.unigram_count.values())
        vocab_size = len(self.unigram_count)
        # Total number of characters in the corpus
        total_chars = sum(self.unigram_count.values())
        # Calculate smoothed probabilities using Add-1 smoothing
        next_char_probabilities = {char: (count + 1) / (total_chars + vocab_size) for char, count in self.unigram_count.items()}

## Please do not change anything in this code block.

smoothed_unigram_model = SmoothedUnigramModel(train_text)

# Check the perplexity of the smoothed unigram model on the train set
print('smoothed unigram train perplexity:',
      smoothed_unigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=smoothed_unigram_model, ngram=1, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences,  num_names=5)

# Release models we don't need any more.
del unigram_model
del smoothed_unigram_model

"""## 1.2 Bigram"""

"""
Implementation of a Bigram Model.
"""

class BigramModel(NGramLanguageModel):
    def __init__(self, train_text):
        """
        Initialise and train the model with train_text.

        Args:
            train_text [list of list]: list of tokenised names
        """
        self.bigram_count=get_bigram_counts(train_text)

        # ADD YOUR CODE HERE

    def get_next_char_probabilities(self):
        """
        Returns a probability distribution over all chars in the vocabulary.
        Probability distribution should sum to one.

        Returns:
            P: dictionary or nested dictionary; Output format depends on n-gram
            Examples:
                for N=2 (bigram); dict[key:bigram_char1, value:dict[key:bigram_char2,value:probability of bigram]]
                    > P["a"]["b"] = 0.0001 (which stands of P("b"|"a"))
        """
        next_char_probabilities = {}

        for char1, char_count in self.bigram_count.items():
            total_counts = sum(char_count.values())
            if total_counts == 0:
                next_char_probabilities[char1] = {}  # Handle the case when there are no bigrams for char1
                continue
            next_char_probabilities[char1] = {}
            for char2, count in char_count.items():
                prob = count / total_counts
                next_char_probabilities[char1][char2] = prob
            # ADD YOUR CODE HERE

        return next_char_probabilities


    def get_name_log_probability(self, name):
        """
        Calculates the log probability of name according to the n-gram model.
        Be careful with cases for which probability of the name is zero.

        Args:
            name [list]: list of tokens

        Returns:
            Log probability of the name [float]
        """
        # name_log_probability=0
        #   for char in name:
        #   name_log_probability += np.log(self.unigram_count.get(char,0)/self.unigram_total)
        #   return name_log_probability
        # ADD YOUR CODE HERE

        name_log_probability=0.0
        for i in range(1,len(name)):
            char1, char2=name[i-1], name[i]
            if char1 in self.bigram_count and char2 in self.bigram_count[char1]:
                prob=self.bigram_count[char1].get(char2,0)/sum(self.bigram_count[char1].values())
                name_log_probability+=np.log(prob)
            else:
                name_log_probability +=np.log(1e-10)
        return name_log_probability


    def get_perplexity(self, text):
        """
        Returns the perplexity of the model on a text as a float.

        Args:
            text [list]: a list of string tokens

        Returns:
            perplexity of the given text [float]
        """

        # ADD YOUR CODE HERE
        log_prob = sum(self.get_name_log_probability(name)/len(name) for name in text)
        # Calculate perplexity
        perplexity = math.exp(-log_prob / len(text))
        return perplexity



    def generate_names(self, k, n=MAX_NAME_LENGTH, prefix=None):
        """
        Given a prefix, generate k names according to the model.
        The default prefix is None.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated

        Returns:
            list of generated names [list]
        """
        # ADD YOUR CODE HERE

        names = []
        next_char_probs = self.get_next_char_probabilities()
        for _ in range(k):
            name = []
            if prefix:
                name.extend(prefix)
                current_char=prefix[-1]
            else:
                current_char = '<s>'

            while current_char !=  '<\s>' and len(name) < n :
                if current_char in self.bigram_count:
                    next_char = random.choices(list(next_char_probs[current_char].keys()), weights=list(next_char_probs[current_char].values()))[0]
                    name.append(next_char)
                    current_char=next_char
                else:
                    break
            name_str = "".join(name)
            names.append(name_str)

        return names


    def get_most_likely_chars(self, sequence, k):
        """
        Given a sequence of characters, outputs k most likely characters after the sequence.

        Args:
            sequence [list[str]]: list of characters
            k [int]: number of most likely characters to return

        Returns:
            chars [list[str]]: *Ordered* list of most likely characters
                        (with charcater at index 0 being the most likely and
                        character at index k-1 being the least likely)

        """
        # ADD YOUR CODE HERE
        most_likely_chars=[]
        if len(sequence)>0 and sequence[-1] in self.bigram_count:
            next_char_probs = self.bigram_count[sequence[-1]]
            char = sorted(next_char_probs.keys(), key=lambda x: next_char_probs[x], reverse=True)
            most_likely_chars = char[:k]
        return most_likely_chars

"""### Eval"""

## Please do not change anything in this code block.

bigram_model = BigramModel(train_text)

# check the perplexity of the bigram model on training data
print('bigram train perplexity:',
      bigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=bigram_model, ngram=2, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

"""### Smoothing

Implement a smoothed version of the bigram model. You may extend the `BigramModel` class and re-use some of the functions.

You will implement the following smoothing techniques:
-  Laplace or add-k smoothing
- Interpolation

**Laplace or Add-k smoothing**
- what is the effect of changing `k`?
Choosing the right hyperparameters can significantly impact the performance of the language model. Let's consider the hyperparameter BIGRAM_LAPLACE_K, which is used for add-k or Laplace smoothing in bigram models.

In add-k or Laplace smoothing, we add a constant value (k) to all observed counts to handle unseen events. Choosing an appropriate value for k is crucial. A larger value of k leads to more smoothing, which can result in less overfitting but potentially less accurate modeling of the data. On the other hand, a smaller value of k may lead to more overfitting but potentially more accurate modeling of the training data.

To choose the hyperparameter BIGRAM_LAPLACE_K, we can experiment with different values and observe the performance metrics such as the quality of generated names and the perplexity of the model on the validation set. Here's how you can set the hyperparameter:
"""

"""choose your hyperparameter and see the difference in performance"""

# ADD YOUR CODE HERE

# CHANGE THE None VALUES TO YOUR DESIRED VALUES
# Please feel free to play with these hyperparameters to see the effects on the
# quality of generated names and perplexity

BIGRAM_LAPLACE_K = 0.5 # value of k for add-k or Laplac smoothing in bigram models

"""
Implementation of a bigram model with laplace or add-k smoothing.

"""

class LaplaceSmoothedBigramModel(BigramModel):
    # This class extends BigramModel.

    def __init__(self, train_text, k):
        super().__init__(train_text)
        self.k = BIGRAM_LAPLACE_K # specify k for smoothing

    # You should override ONLY those functions
    # which calculate probability of a bigram.
    # You can override get_next_char_probabilities
    # or any other helper functions you use in BigramModel
    # to calculate bigram probabilities.

    # ADD YOUR CODE HERE
    def get_next_char_probabilities(self):
        """
        Returns a probability distribution over all chars in the vocabulary.
        Probability distribution should sum to one.

        Returns:
            P: dictionary or nested dictionary; Output format depends on n-gram
            Examples:
                for N=2 (bigram); dict[key:bigram_char1, value:dict[key:bigram_char2,value:probability of bigram]]
                    > P["a"]["b"] = 0.0001 (which stands of P("b"|"a"))
        """
        next_char_probabilities = {}

        for char1, char_count in self.bigram_count.items():
            total_counts = sum(char_count.values())
            vocab_size = len(char_count)  # Size of vocabulary for Laplace smoothing
            next_char_probabilities[char1] = {}

            for char2 in char_count.keys():
                count = char_count.get(char2, 0)  # Get count of bigram (char1, char2)
                prob = (count + self.k) / (total_counts + self.k * vocab_size)
                next_char_probabilities[char1][char2] = prob

            # Normalize probabilities to ensure the sum is 1
            sum_probs = sum(next_char_probabilities[char1].values())
            for char2 in next_char_probabilities[char1]:
                next_char_probabilities[char1][char2] /= sum_probs

        return next_char_probabilities

## Please do not change anything in this code block.

smoothed_bigram_model = LaplaceSmoothedBigramModel(train_text, k=BIGRAM_LAPLACE_K)

# check the perplexity of the bigram model on training data
print('smoothed bigram train perplexity:',
      smoothed_bigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=smoothed_bigram_model, ngram=2, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

"""**Interpolation**
- what are good values for `lambdas` in interpolation?
"""

"""choose your hyperparameter and see the difference in performance"""

# ADD YOUR CODE HERE

# CHANGE THE None VALUES TO YOUR DESIRED VALUES
# Please feel free to play with these hyperparameters to see the effects on the
# quality of generated names and perplexity

BIGRAM_LAMBDAS = (0.7,0.3) # lambdas for interpolation smoothing in bigram models

"""
Implementation of a bigram model with interpolation smoothing
"""

class InterpolationSmoothedBigramModel(BigramModel):

    def __init__(self, train_text, lambdas):
        super().__init__(train_text)
        self.lambda_1, self.lambda_2 = 0.7,0.3
        self.unigram_count = get_unigram_counts(train_text)
        # self.unigram_total = sum(self.unigram_count.values())
        self.smoothed_unigram_model = SmoothedUnigramModel(train_text)
        self.unigram_prob= self.smoothed_unigram_model.get_next_char_probabilities()

    # You should override ONLY those functions
    # which calculate probability of a bigram.
    # You can override get_next_char_probabilities
    # or any other helper functions you use in BigramModel
    # to calculate bigram probabilities.

    # ADD YOUR CODE HERE

        """
        Initialise and train the model with train_text.

        Args:
            train_text [list of list]: list of tokenised names
            lambda_1 [float]: Interpolation weight for bigram model
            lambda_2 [float]: Interpolation weight for unigram model
        """
    def get_next_char_probabilities(self):
        """
        Returns a probability distribution over all chars in the vocabulary.
        Probability distribution should sum to one.

        Returns:
            P: dictionary or nested dictionary; Output format depends on n-gram
            Examples:
                for N=2 (bigram); dict[key:bigram_char1, value:dict[key:bigram_char2,value:probability of bigram]]
                    > P["a"]["b"] = 0.0001 (which stands of P("b"|"a"))
        """
        next_char_probabilities = {}
        for char1, char_count in self.bigram_count.items():
            total_counts = sum(char_count.values())
            next_char_probabilities[char1] = {}
            for char2, count in char_count.items():
                bigram_prob = count / total_counts
                unigram_prob = self.unigram_prob[char2]
                prob = self.lambda_1 * bigram_prob + self.lambda_2 * unigram_prob
                next_char_probabilities[char1][char2] = prob

            # Normalize probabilities
            total_prob = sum(next_char_probabilities[char1].values())
            next_char_probabilities[char1] = {char2: prob / total_prob for char2, prob in next_char_probabilities[char1].items()}

        return next_char_probabilities

    def get_unigram_probabilities(self):
        """
        Returns the unigram probabilities.

        Returns:
            P: dictionary; Unigram probabilities
            Example:
                > P["a"] = 0.001 (which stands of P("a"))
        """
        unigram_probabilities = {}
        for char, count in self.unigram_count.items():
            prob = count / self.unigram_total
            unigram_probabilities[char] = prob
        return unigram_probabilities

## Please do not change anything in this code block.

smoothed_bigram_model = InterpolationSmoothedBigramModel(train_text, lambdas=BIGRAM_LAMBDAS)

# check the perplexity of the bigram model on training data
print('smoothed bigram train perplexity:',
      smoothed_bigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=smoothed_bigram_model, ngram=2, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

# Release models we don't need any more.
del bigram_model
del smoothed_bigram_model

"""## 1.3 Trigram (smoothed)"""

"""choose your hyperparameter and see the difference in performance"""

# ADD YOUR CODE HERE

# CHANGE THE None VALUES TO YOUR DESIRED VALUES
# Please feel free to play with these hyperparameters to see the effects on the
# quality of generated names and perplexity

TRIGRAM_LAMBDAS = (0.6, 0.25, 0.15) # lambdas for interpolation smoothing in trigram models



"""
Implementaion of a Trigram Model with interpolation smoothing.
"""

class TrigramModel(NGramLanguageModel):
    def __init__(self, train_text):
        """
        Initialise and train the model with train_text.

        Args:
            train_text [list of list]: list of tokenised names
        """
        # ADD YOUR CODE HERE
        self.trigram_count=get_trigram_counts(train_text)
        self.lambda_1, self.lambda_2,self.lambda_3 = 0.6,0.25,0.15

        self.unigram_count = get_unigram_counts(train_text)
        # self.unigram_total = sum(self.unigram_count.values())
        self.bigram_count = get_bigram_counts(train_text)
        # self.bigram_total = sum(self.bigram_count.values())
        self.smoothed_bigram_model = InterpolationSmoothedBigramModel(train_text, lambdas=BIGRAM_LAMBDAS)
        self.bigram_prob=self.smoothed_bigram_model.get_next_char_probabilities()
        self.smoothed_unigram_model = SmoothedUnigramModel(train_text+vocab.get_itos())
        self.unigram_prob= self.smoothed_unigram_model.get_next_char_probabilities()
        self.unseen_trigram_prob = 1e-8  # A small non-zero value
        self.vocab=vocab.get_stoi()
    def get_next_char_probabilities(self):
        """
        Returns a probability distribution over all chars in the vocabulary.
        Probability distribution should sum to one.

        Returns:
            P: dictionary or nested dictionary; Output format depends on n-gram
            Examples:
                for N=1 (unigram); dict[key:unigram,value:probability of unigram]
                    > P["a"] = 0.0001
                for N=2 (bigram); dict[key:bigram_char1, value:dict[key:bigram_char2,value:probability of bigram]]
                    > P["a"]["b"] = 0.0001 (corresponding to P(b|a))
                for N=3 (trigram); dict[dict[dict]]
                    > P["a"]["b"]["c"] = 0.0001 (corresponding to P(c|ab))
        """

        # ADD YOUR CODE HERE
        next_char_probabilities = {}

        for char1 in self.vocab:
          next_char_probabilities[char1]={}
          for char2 in self.vocab:
            next_char_probabilities[char1][char2]={}
            for char3 in self.vocab:
              next_char_probabilities[char1][char2][char3]=self.unseen_trigram_prob

        for char1,char1_value in self.trigram_count.items():
            # next_char_probabilities[char1]={}
            for char2,char2_value in char1_value.items():
                total_count=sum(char2_value.values())
                # next_char_probabilities[char1][char2]={}
                for char3,count in char2_value.items():
                  unigram_prob = self.unigram_prob.get(char3, 1e-8)
                  bigram_prob = self.bigram_prob.get(char2, {}).get(char3, 1e-8)
                  trigram_prob = count / total_count
                  next_char_probabilities[char1][char2][char3]= self.lambda_1*trigram_prob + self.lambda_2*bigram_prob + self.lambda_3*unigram_prob


                total_prob = sum(next_char_probabilities[char1][char2].values())
                next_char_probabilities[char1][char2] = {char3: prob / total_prob for char3, prob in next_char_probabilities[char1][char2].items()}

        return next_char_probabilities
        # next_char_probabilities = {}
        # for char1,char1_value in self.trigram_count.items():
        #     next_char_probabilities[char1]={}
        #     for char2,char2_value in self.trigram_count() :
        #         total_count=sum(char2_value.values())
        #         next_char_probabilities[char1][char2]={}
        #         for char3,count in char2_value.items():
        #           unigram_prob = self.unigram_prob.get(char3, 1e-8)
        #           bigram_prob = self.bigram_prob.get(char2, {}).get(char3, 1e-8)
        #           trigram_prob = count / total_count
        #           next_char_probabilities[char1][char2][char3]= self.lambda_1*trigram_prob + self.lambda_2*bigram_prob + self.lambda_3*unigram_prob


        #         total_prob = sum(next_char_probabilities[char1][char2].values())
        #         next_char_probabilities[char1][char2] = {char3: prob / total_prob for char3, prob in next_char_probabilities[char1][char2].items()}



    def get_name_log_probability(self, name):
        """
        Calculates the log probability of name according to the n-gram model.
        Be careful with cases for which probability of the name is zero.

        Args:
            name [list]: list of tokens

        Returns:
            Log probability of the name [float]
        for bigram



        """

        log_probability=0.0
        next_char_prob=self.get_next_char_probabilities()
        for i in range(2,len(name)):
          char1, char2, char3 =name[i-2], name[i-1], name[i]
          prob=next_char_prob[char1][char2][char3]
          log_probability += np.log(prob)
        name_log_probability=log_probability
        return name_log_probability


    def get_perplexity(self, text):
        """
        Returns the perplexity of the model on a text as a float.

        Args:
            text [list]: a list of string tokens

        Returns:
            perplexity of the given text [float]
        """

        # ADD YOUR CODE HERE
        log_prob = sum(self.get_name_log_probability(name)/len(name) for name in text)
        # Calculate perplexity
        perplexity = math.exp(-log_prob / len(text))
        return perplexity


    def generate_names(self, k, n=MAX_NAME_LENGTH, prefix=None):
        """
        Given a prefix, generate k names according to the model.
        The default prefix is None.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated

        Returns:
            list of generated names [list]

                current_char = prefix[-1]
                previous_char = prefix[-2]
            else:
                current_char = self.START
                previous_char = self.START

            while current_char != self.END and len(name) < n:
                if previous_char in next_char_probs and current_char in next_char_probs[previous_char]:
                    char_probs = next_char_probs[previous_char][current_char]
                    next_char = random.choices(list(char_probs.keys()), weights=list(char_probs.values()))[0]
                    name.append(next_char)
                    previous_char = current_char
                    current_char = next_char
                else:
                    break
            name_str = "".join(name)
            names.append(name_str)
        return names
        """
        # ADD YOUR CODE HERE
        names = []
        next_char_probs = self.get_next_char_probabilities()
        for _ in range(k):
          name= []
          if prefix:
            name.extend(prefix)
            current_char=prefix[-1]
            previous_char=prefix[-2]
          else:
            current_char = START
            previous_char = START

          while current_char != END and len(name) < n:
            if previous_char in next_char_probs and current_char in next_char_probs[previous_char]:
              char_probs = next_char_probs[previous_char][current_char]
              next_char = random.choices(list(char_probs.keys()), weights=list(char_probs.values()))[0]
              name.append(next_char)
              previous_char = current_char
              current_char = next_char
            else:
              break
            name_str = "".join(name)
            names.append(name_str)

          name_str = "".join(name)
          names.append(name_str)


        return names


    def get_most_likely_chars(self, sequence, k):
        """
        Given a sequence of characters, outputs k most likely characters after the sequence.

        Args:
            sequence [list[str]]: list of characters
            k [int]: number of most likely characters to return

        Returns:
            chars [list[str]]: *Ordered* list of most likely characters
                        (with charcater at index 0 being the most likely and
                        character at index k-1 being the least likely)

        """
        # ADD YOUR CODE HERE
        most_likely_chars=[]
        return most_likely_chars

"""#### Eval"""

## Please do not change anything in this code block.

trigram_model = TrigramModel(train_text)

print('trigram train perplexity:',
      trigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=trigram_model, ngram=3, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

# Release models we don't need any more.
del trigram_model

"""# Module 2: Neural Language Modelling

## 2.1 Neural N-gram Language Model

For this part of the assignment, you should use the GPU (you can do this by changing the runtime of this notebook).

In this section, you will implement a neural version of an n-gram model.  The model will use a simple feedforward neural network that takes the previous `n-1` chars and outputs a distribution over the next char.

You will use PyTorch to implement the model.  We've provided a little bit of code to help with the data loading using [PyTorch's data loaders](https://pytorch.org/docs/stable/data.html)
"""

# Import the necessary libraries

import math
import time
import random
import os, sys
import json
from functools import partial

from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim

from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F

## Please do not change anything in this code block.

def collate_ngram(batch, text_pipeline):
    """
    Converts the text in the batch to tokens
    and maps the tokens to indices in the vocab.
    The text in the batch is a list of ngrams
    i.e. if N=3, then text contains 3 tokens in a list
    and batch is a list of such texts.

    Returns:
        batch_input [pytorch tensor]:
            input for n-gram model with size batch_size*(ngram-1)
        batch_output [pytorch tensor]:
            output for n-gram model with size batch_size
    """

    batch_input, batch_output = [], []

    # Process each text in the batch
    for text in batch:
        token_id_sequence = text_pipeline(text)
        # last token is the output, and
        #  pervious ngram-1 tokens are inputs
        output = token_id_sequence.pop()
        input = token_id_sequence
        batch_input.append(input)
        batch_output.append(output)

    # Convert lists to PyTorch tensors and moves to the gpu (if using)
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    if USE_CUDA:
        batch_input = batch_input.cuda()
        batch_output = batch_output.cuda()

    return batch_input, batch_output


def get_dataloader(input_text, vocab, ngram, batch_size, shuffle):
    """
    Creates a dataloader for the n-gram model which
    takes in a list of list of tokens, appends the START token
    at the starting of each text, and converts text into ngrams.

    Example: For a trigram model, the list of characters are
        ["n", "a", "v", "r"]
    will be converted into lists
        ["n", "a", "v"], ["a", "v", "r"]

    For each ngram, first ngram-1 tokens are input and last token
    is the output. Each token is converted into a index in the vocab.
    The dataloader generates a batch of input, output pairs as
    pytorch tensors.


    Args:
        input_text [list[list[str]]]: list of list of tokens
        vocab [torchtext.vocab]: vocabulary of the corpus
    """

    ngram_sequences = []
    for text in input_text:
        if text[0] == START:
            text = [START]*(N_GRAM_LENGTH-2) + text
        else:
            text = [START]*(N_GRAM_LENGTH-1) + text

        # Create training pairs for each char in the text
        for idx in range(len(text) - ngram + 1):
            ngram_sequence = text[idx : (idx + ngram)]
            ngram_sequences.append(ngram_sequence)

    text_pipeline = lambda x: vocab(x)
    collate_fn = collate_ngram

    # creates a DataLoader for the dataset

    """
    dataloader documentation
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    dataloader = DataLoader(
        ngram_sequences,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
        )
    return dataloader

"""#### FNN Implementation

**Feed-forward Neural Language Modelling**

Like the n-gram LM, the feedforward neural LM approximates the probability of a char given the entire prior context $P(w_t|w_{1:t−1})$ by approximating based on the $N-1$ previous chars:
$$P(w_t|w_1,...,w_{t−1}) ≈ P(w_t|w_{t−N+1},...,w_{t−1})$$


Implement the FNN LM given in this paper: [Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

The architecture of the FNN can be described by the equation and figure:

$$y = b + W x + U \text t \text a \text n \text h (d + H x)$$

- $x$ is of size $(ngram-1)*m$ where $m$ is the size embedding dimensions
- $y$ is of size $V*1$ where $V$ is the vocabulary size

![FNN_LM](https://drive.google.com/uc?id=1aQhkXjWelHfiBfmBQV3z5TjHFNMtqtzT)


**Some tips**:
- embed the chars with dimension $m$ (example, $60$), then flatten into a single embedding for  $n-1$  chars (with size  $(n-1)*m$ )
- you can use Adam or Stochastic Gradient Descent (SGD) for optimising the cross entropy loss
- If you are using SGD, you may want to use momentum, and a learning rate scheduler
- do early stopping based on validation set loss or perplexity

**Important**: Fix seed as 42 whenever performing any randomized operations, e.g., initializing ML models.
"""

"""
Implemenation of a PyTorch Module that holds the neural network for your model

"""
class FNN_LM(nn.Module):

    def __init__(self, vocab_size, emb_size, hid_size, ngram):
        super(FNN_LM, self).__init__()
        self.ngram = ngram
        self.embsize=emb_size
        # YOUR CODE HERE

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # Linear layers
        self.linear1 = nn.Linear((ngram - 1) * emb_size, hid_size)
        self.linear2 = nn.Linear(hid_size,vocab_size)
        self.tanh=nn.Tanh()

        # Move model to CUDA if available
        if USE_CUDA:
            self.cuda()



    def forward(self, chars):
        """
        Args:
            chars: this is a tensor of inputs with shape [batch_size x ngram-1]

        Returns:
            logits: a tensor of log probabilities with shape [batch_size x vocab_size]

        """

        # YOUR CODE HERE
        # Embedding the characters
        x = self.embedding(chars)  # [batch_size, ngram-1, emb_size]
        # print("Shape of x before view:", x.shape)
        flat= x.view(x.size(0),-1)
        # print("Shape of x after view:", x.shape)



        u=self.linear1(flat)
        hidden=self.tanh(u)
        # Linear layers
        logits=self.linear2(hidden)

        return logits

"""**The following is the Trainer class for the FNN LM. Add your code for the `training` and `validation` loops.**"""

class NeuralNGramTrainer:
    """
    NeuralNGramTrainer wraps FNN_LM to handle training and evaluation.
    """
    # NOTE: you are free to add additional inputs/functions
    # to NeuralNGramTrainer to make training better
    # make sure to define and add it within the input
    # and initialization if you are using any additional inputs
    # for usage in the function

    def __init__(
        self,
        ngram,
        model,
        optimizer,
        criterion,
        train_dataloader,
        valid_dataloader,
        epochs,
        use_cuda,
        vocab,
        model_dir
    ):

        self.ngram = ngram
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.use_cuda = use_cuda
        self.model_dir = model_dir
        self.loss = {"train": [], "val": []}
        self.vocab = vocab

        # Move the model to GPU if available
        if self.use_cuda:
            self.model = self.model.cuda()


    def train(self):
        """
        Trains the model with train_dataloader and validates using valid_dataloader

        """
        # You may change the input arguments to this function,
        # but make sure to also change the code wherever this function is called

        # ADD YOUR CODE HERE
        # FOR TRAINING & VALIDATION
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch_input, batch_output in self.train_dataloader:
                if self.use_cuda:
                    batch_input, batch_output = batch_input.cuda(), batch_output.cuda()
                self.optimizer.zero_grad()
                output = self.model(batch_input)
                loss = self.criterion(output, batch_output)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss =total_loss/ len(self.train_dataloader)
            self.loss['train'].append(avg_train_loss)


            self.model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                for batch_input, batch_output in self.valid_dataloader:
                    if self.use_cuda:
                        batch_input, batch_output = batch_input.cuda(), batch_output.cuda()
                    output = self.model(batch_input)
                    loss = self.criterion(output, batch_output)

                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss/len(self.valid_dataloader)
                self.loss['val'].append(avg_val_loss)

            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')




    def plot_losses(self):
        """
        Plots the training and validation losses
        """
        plt.plot(self.loss['train'], label='train_ppl')
        plt.plot(self.loss['val'], label='val_ppl')
        plt.legend()
        plt.show


    def save_model(self):
        """
        Save final model to directory

        """

        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)


    def save_loss(self):
        """
        Save train/val loss as json file to the directory

        """

        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)


    def get_next_char_probabilities(self):
        """
        Return a dictionary of probabilities for each char in the vocabulary
        with a default starting sequence of [START]*(ngram-1)
        Example:
            If ngram=3, then default starting sequence for which
            probabilities have to be returned is
            [START, START]

        Returns:
            dictionary with key: char, value: probability

        """

        # ADD YOUR CODE HERE
        start_sequence = [START] * (self.ngram - 1)

        char_probs = {}
        with torch.no_grad():
          self.model.eval()
          prefix_indices = torch.tensor([[self.vocab[token] for token in start_sequence]], dtype=torch.long)
          if self.use_cuda:
            prefix_indices = prefix_indices.cuda()
          output = self.model(prefix_indices)
          itos = vocab.get_itos()
          probabilities = torch.softmax(output[0, -1], dim=0)
          for token_index, probability in enumerate(probabilities.unsqueeze(0)):
            char_probs[itos[token_index]] = probability.item()

        return char_probs




    def generate_names(self, k, n=MAX_NAME_LENGTH, prefix=None):
        """
        Given a prefix, generate k names according to the model.
        The default prefix is None.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated

        Returns:
            list of generated names [list[str]]



        sequence=sequence.copy()
        while (len(sequence)) <self.ngram-1:
          sequence.insert(0,START)


        with torch.no_grad():
          input_sequence = [self.vocab.get_stoi()[token] for token in sequence][-(self.ngram-1):]
          input_tensor = torch.tensor([input_sequence])
        """

        # ADD YOUR CODE HERE

        # don't forget self.model.eval()

        generated_names = []
        self.model.eval()

        with torch.no_grad():
          for _ in range(k):
            current_sequence = [START] * (self.ngram - 1)  # Default starting sequence
            if prefix is not None:
                # Ensure that prefix has the proper dimension
                current_sequence.extend(prefix)
                current_sequence = current_sequence[-(self.ngram - 1):]
            name = ""
            for i in current_sequence:
              if i is START:
                continue
              name += i
            for i in range(n):

                input_tensor = torch.tensor([self.vocab[token] for token in current_sequence]).unsqueeze(0)
                if self.use_cuda:
                    input_tensor = input_tensor.cuda()
                output_probs = F.softmax(self.model(input_tensor), dim=-1).squeeze(0)
                next_token_index = torch.multinomial(output_probs, 1).item()
                lookup_index=self.vocab.get_itos()
                next_token =lookup_index[next_token_index]
                if next_token == END:
                  break
                current_sequence = current_sequence[1:] + [next_token]

                name += next_token
            name_str = "".join(name)
            generated_names.append(name_str)


          return generated_names



    def get_perplexity(self, text):
        """
        Returns the perplexity of the model on text as a float.

        Args:
            text [list[list[str]]]: list of tokenised names
            > Example:
            [['<s>', 'a', 'a', 'b', 'i', 'd', '</s>'],
            ['<s>', 'a', 'a', 'b', 'i', 'd', 'a', '</s>']]

        Returns:
            perplexity [float]

        """

        # ADD YOUR CODE HERE

        # you may want to use the dataloader here
        # don't forget self.model.eval()
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_tokens = 0


        dataloader = get_dataloader(text,vocab,ngram = N_GRAM_LENGTH, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

        with torch.no_grad():
          for batch_input, batch_output in dataloader:
            output = self.model(batch_input)
            batch_output = batch_output.view(-1)
            loss = F.cross_entropy(output, batch_output)
            total_loss += loss.item() * batch_output.size(0)  # Multiply by batch size to account for varying token lengths
            total_tokens += batch_output.size(0)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity



    # def get_most_likely_chars(self, sequence, k):
    #     """
    #     Given a sequence of characters, outputs k most likely characters after the sequence.

    #     Args:
    #         sequence [list[str]]: list of characters
    #         k [int]: number of most likely characters to return

    #     Returns:
    #         chars [list[str]]: *Ordered* list of most likely characters
    #                     (with charcater at index 0 being the most likely and
    #                     character at index k-1 being the least likely)
    #     """

    #     # ADD YOUR CODE HERE

    #     # don't forget self.model.eval()

    #     self.model.eval()
    #     sequence=sequence.copy()
    #     while (len(sequence)) <self.ngram-1:
    #       sequence.insert(0,START)


    #     with torch.no_grad():
    #       input_sequence = [self.vocab.get_stoi()[token] for token in sequence][-(self.ngram-1):]
    #       input_tensor = torch.tensor([input_sequence])
    #       if self.use_cuda:
    #         input_tensor = input_tensor.cuda()
    #       output_probs = F.softmax(self.model(input_tensor), dim=-1).squeeze(0)[-1]  # Get probabilities for the next character
    #       if output_probs.dim() == 0:  # Check if output_probs is a scalar tensor
    #         output_probs = output_probs.unsqueeze(0)  # Convert to a tensor with a single element
    #       if len(output_probs) == 0:
    #         return []  # Return an empty list if no probabilities are available
    #       sorted_indices = output_probs.argsort(descending=True)
    #       lookup_index=self.vocab.get_itos()
    #       chars = [lookup_index[index] for index in sorted_indices[:k]]
    #     return chars
    def get_most_likely_chars(self, sequence, k):
            """
            Given a sequence of characters, outputs k most likely characters after the sequence.

            Args:
                sequence [list[str]]: list of characters
                k [int]: number of most likely characters to return

            Returns:
                chars [list[str]]: Ordered list of most likely characters
                            (with charcater at index 0 being the most likely and
                            character at index k-1 being the least likely)

            """

            # ADD YOUR CODE HERE

            # don't forget self.model.eval()

            # BEGIN CODE
            # Ensure model is in evaluation mode
            self.model.eval()

            # Ensure sequence is of appropriate length
            while len(sequence) < self.ngram - 1:
                sequence.insert(0, START)

            most_likely_chars = []

            # Prepare current sequence
            current_sequence = [self.vocab.get_stoi().get(char) for char in sequence][-self.ngram + 1:]
            current_sequence_tensor = torch.tensor([current_sequence])

            # Move tensor to GPU if available
            if self.use_cuda:
                current_sequence_tensor = current_sequence_tensor.cuda()

            # Get logits from model
            logits = self.model(current_sequence_tensor)

            # Calculate probabilities and retrieve top k indices
            probabilities = torch.softmax(logits, dim=1).squeeze()
            topk_values, topk_indices = torch.topk(probabilities, k)

            # Retrieve characters corresponding to top k indices
            for idx in topk_indices:
                most_likely_chars.append(self.vocab.get_itos()[idx])

            return most_likely_chars

"""choose your hyperparameter and see the difference in performance"""

# ADD YOUR CODE HERE

# CHANGE THE None VALUES TO YOUR DESIRED VALUES
# Please feel free to play with these hyperparameters to see the effects on the
# quality of generated names and perplexity

MAX_NAME_LENGTH = 12 # maximum length of name for generation

# Remember to fix seed as 42
torch.manual_seed(42)

# check if GPU is available
USE_CUDA = torch.cuda.is_available()
print(f"GPU is available: {USE_CUDA}")

N_GRAM_LENGTH = 7 # The length of the n-gram (N_GRAM_LENGTH=3 for trigram)
EMB_SIZE = 64 # The size of the embedding
HID_SIZE = 256 # The size of the hidden layer
EPOCHS = 5
BATCH_SIZE = 128
SHUFFLE = True # if dataset should be shuffled

## Please do not change anything in this code block.

# Get data iterator and build vocabulary from input text
train_text, vocab = get_tokenised_text_and_vocab(ds_type='train')
validation_text, _ = get_tokenised_text_and_vocab(ds_type='valid', vocab=vocab)

# Check the size of vocabulary
vocab_size = len(vocab.get_stoi())
print(vocab_size)

# Load training and validation dataloaders
train_dataloader = get_dataloader(train_text, vocab, ngram = N_GRAM_LENGTH, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
valid_dataloader = get_dataloader(validation_text, vocab, ngram = N_GRAM_LENGTH, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

# ADD YOUR CODE HERE

# This is the part where you should train your FNN_LM model

# CHANGE THE None VALUES TO YOUR DESIRED VALUES

# Initialise the model, optimizer, learning rate scheduler (optional), and loss criteria
model = FNN_LM(vocab_size=vocab_size, emb_size=EMB_SIZE, hid_size=HID_SIZE, ngram=N_GRAM_LENGTH)
# Move the model to GPU if available
if USE_CUDA:
  model = model.cuda()


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ADD YOUR CODE HERE
# change the directory name with your SAPname and SRno

model_dir = 'SAPname_SRno/fnn'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# NOTE: if you are **optionally** using additional options for the trainer
# (e.g., a training scheduler), please add them below.
trainer = NeuralNGramTrainer(
        ngram=N_GRAM_LENGTH,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        epochs=EPOCHS,
        use_cuda=USE_CUDA,
        model_dir=model_dir,
        vocab=vocab)

# Train the model
trainer.train()
print("Training finished.")

trainer.save_model()
trainer.save_loss()
vocab_path = os.path.join(model_dir, "vocab.pt")
torch.save(vocab, vocab_path)
print("Model artifacts saved to folder:", model_dir)

"""### Eval"""

eval_ngram_model(trainer, ngram=N_GRAM_LENGTH, ds=validation_text, ds_name='valid', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5, is_neural=True)

"""Load your saved model and generate a few names"""

START = "<s>"   # Start-of-name token
END = "</s>"    # End-of-name token
UNK = "<unk>"   # token representing out of unknown (or out of vocabulary) tokens

# ADD YOUR CODE HERE
# change the directory name with your SAPname and SRno

folder = 'SAPname_SRno/fnn'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the saved model
model = torch.load(f"{folder}/model.pt", map_location=device)
vocab = torch.load(f"{folder}/vocab.pt")

# NOTE: if you are **optionally** using additional options for the trainer
# (e.g., a training scheduler), please add them below.
trainer = NeuralNGramTrainer(
        ngram=N_GRAM_LENGTH,
        model=model,
        optimizer=None,
        criterion=None,
        train_dataloader=None,
        valid_dataloader=None,
        epochs=None,
        use_cuda=USE_CUDA,
        model_dir=None,
        vocab=vocab)

# Generate a few names
names = trainer.generate_names(k=5, n=MAX_NAME_LENGTH, prefix=['a','s','h'])
print(", ".join(names))

# you may use this block to test if your model and vocab load properly,
# and that your functions are able to generate sentences, calculate perplexity etc.

# Release models we don't need any more.
del trainer
del model

"""## 2.2 Recurrent Neural Networks for Language Modelling

For this stage of the assignment, you will implement an RNN language model.

Some tips:
* use dropout
* use the same weights for the embedding layer and the pre-softmax layer
* train with Adam
"""

"""
Implemenation of a PyTorch Module that holds the RNN

"""
class RNN_LM(nn.Module):

    # you may change the input arguments for __init__
    def __init__(self):
        super(RNN_LM, self).__init__()

        # YOUR CODE HERE

    def forward(self):

        # YOUR CODE HERE

        return

class RNNTrainer:
    """
    RNNTrainer wraps RNN_LM to handle training and evaluation.

    """

    # NOTE: you are free to add additional inputs/functions
    # to RNNTrainer to make training better
    # make sure to define and add it within the input
    # and initialization if you are using any additional inputs
    # for usage in the function

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_dataloader,
        valid_dataloader,
        epochs,
        use_cuda,
        vocab,
        model_dir
    ):

        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.use_cuda = use_cuda
        self.model_dir = model_dir
        self.loss = {"train": [], "val": []}
        self.vocab = vocab

        # Move the model to GPU if available
        if self.use_cuda:
            self.model = self.model.cuda()

    def train(self):
        """
        Trains the model with train_dataloader and validates using valid_dataloader

        """
        # You may change the input arguments to this function,
        # but make sure to also change the code wherever this function is called

        # ADD YOUR CODE HERE FOR TRAINING & VALIDATION


    def save_model(self):
        """
        Save final model to directory

        """

        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)


    def save_loss(self):
        """
        Save train/val loss as json file to the directory

        """

        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)


    def get_next_char_probabilities(self):
        """
        Return a dictionary of probabilities for each char in the vocabulary
        with a default starting sequence of [START]

        Returns:
            dictionary with key: char, value: probability

        """

        # ADD YOUR CODE HERE

        return next_char_probabilities


    def generate_names(self, k, n, prefix=None):
        """
        Given a prefix, generate k names according to the model.
        The default prefix is None.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated

        Returns:
            list of generated names [list[str]]
        """

        # ADD YOUR CODE HERE

        # don't forget self.model.eval()

        return names


    def get_perplexity(self, text):
        """
        Returns the perplexity of the model on text as a float.

        Args:
            text [list[list[str]]]: list of tokenised names
            > Example:
            [['<s>', 'a', 'a', 'b', 'i', 'd', '</s>'],
            ['<s>', 'a', 'a', 'b', 'i', 'd', 'a', '</s>']]

        Returns:
            perplexity [float]

        """

        # ADD YOUR CODE HERE

        # you may want to use the dataloader here
        # don't forget self.model.eval()

        return perplexity


    def get_most_likely_chars(self, sequence, k):
        """
        Given a sequence of characters, outputs k most likely characters after the sequence.

        Args:
            sequence [list[str]]: list of characters
            k [int]: number of most likely characters to return

        Returns:
            chars [list[str]]: *Ordered* list of most likely characters
                        (with charcater at index 0 being the most likely and
                        character at index k-1 being the least likely)

        """

        # ADD YOUR CODE HERE

        # don't forget self.model.eval()

        return most_likely_chars

"""choose your hyperparameter and see the difference in performance"""

# ADD YOUR CODE HERE

# CHANGE THE None VALUES TO YOUR DESIRED VALUES
# Please feel free to play with these hyperparameters to see the effects on the
# quality of generated names and perplexity

MAX_NAME_LENGTH = None # maximum length of name for generation

# Remember to fix seed as 42
torch.manual_seed(42)

# check if GPU is available
USE_CUDA = torch.cuda.is_available()
print(f"GPU is available: {USE_CUDA}")

EPOCHS = None
BATCH_SIZE = None
SHUFFLE = None # if dataset should be shuffled

# Get data iterator and build vocabulary from input text
train_text, vocab = get_tokenised_text_and_vocab(ds_type='train')
validation_text, _ = get_tokenised_text_and_vocab(ds_type='valid', vocab=vocab)

# Check the size of vocabulary
vocab_size = len(vocab.get_stoi())
print(vocab_size)

# create the dataloaders for training and validation

# ADD YOUR CODE HERE

train_dataloader = None
valid_dataloader = None

# ADD YOUR CODE HERE

# CHANGE THE None VALUES TO YOUR DESIRED VALUES

# Initialize the model
# you may want to pass arguments to RNN_LM based on your implementation
model = RNN_LM()
# Move the model to GPU if available
if USE_CUDA:
  model = model.cuda()

# Initialise the optimizer, learning rate scheduler (optional), and loss criteria
optimizer = None
criterion = None

# ADD YOUR CODE HERE
# change the directory name with your SAPname and SRno

model_dir = 'SAPname_SRno/rnn'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# NOTE: if you are **optionally** using additional options for the trainer
# (e.g., a training scheduler), please add them below.
trainer = RNNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        epochs=EPOCHS,
        use_cuda=USE_CUDA,
        vocab=vocab,
        model_dir=model_dir
        )

# Train the model
trainer.train()
print("Training finished.")

trainer.save_model()
trainer.save_loss()
vocab_path = os.path.join(model_dir, "vocab.pt")
torch.save(vocab, vocab_path)
print("Model artifacts saved to folder:", model_dir)

"""### Eval"""

## Please do not change anything in this code block.

def eval_rnn_model(model, ds, ds_name, eval_prefixes, eval_sequences, num_names=5):
    """
    Runs the following evaluations on n-gram models:
    (1) checks if probability distribution returned by model.get_next_char_probabilities() sums to one
    (2) checks the perplexity of the model
    (3) generates names using model.generate_names()
    (4) generates names given a prefix using model.generate_names()
    (4) output most likely characters after a given sequence of chars using model.get_most_likely_chars()
    """

    # (1) checks if probability distributions sum to one
    is_valid = check_validity(model, 1, True)
    print(f'EVALUATION probability distribution is valid: {is_valid}')

    # (2) evaluate the perplexity of the model on the dataset
    print(f'EVALUATION of RNN on {ds_name} perplexity:',
        model.get_perplexity(ds))

    # (3) generate a few names
    generated_names = ", ".join(model.generate_names(k=num_names, n=MAX_NAME_LENGTH))
    print(f'EVALUATION RNN generated names are {generated_names}')

    # (4) generate a few names given a prefix
    for prefix in eval_prefixes:
        generated_names_with_prefix = ", ".join(model.generate_names(k=num_names, n=MAX_NAME_LENGTH, prefix=prefix))
        prefix = ''.join(prefix)
        print(f'EVALUATION RNN generated names with prefix {prefix} are {generated_names_with_prefix}')

    # (5) get most likely characters after a sequence
    for sequence in eval_sequences:
        most_likely_chars = ", ".join(model.get_most_likely_chars(sequence=sequence, k=num_names))
        sequence = "".join(sequence)
        print(f"EVALUATION RNN the top most likely chars after {sequence} are {most_likely_chars}")

eval_rnn_model(trainer, ds=validation_text, ds_name='valid', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

START = "<s>"   # Start-of-name token
END = "</s>"    # End-of-name token
UNK = "<unk>"   # token representing out of unknown (or out of vocabulary) tokens

# ADD YOUR CODE HERE
# change the directory name with your SAPname and SRno

folder = 'SAPname_SRno/rnn'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the saved model
model = torch.load(f"{folder}/model.pt", map_location=device)
vocab = torch.load(f"{folder}/vocab.pt")

# NOTE: if you are **optionally** using additional options for the trainer
# (e.g., a training scheduler), please add them below.
trainer = RNNTrainer(
        model=model,
        optimizer=None,
        criterion=None,
        train_dataloader=None,
        valid_dataloader=None,
        epochs=None,
        use_cuda=USE_CUDA,
        model_dir=None,
        vocab=vocab)

# Generate a few names
names = trainer.generate_names(k=5, n=MAX_NAME_LENGTH, prefix=['a','a','s','h'])
print(", ".join(names))

# you may use this block to test if your model and vocab load properly,
# and that your functions are able to generate sentences, calculate perplexity etc.

# Release models we don't need any more.
del trainer
del model

"""# Congratulations!

You've reacehd the end of the assignment. Hope this was a pleasant and fun exercise!

Check the submission instructions mentioned at the begining of the notebook in order to submit your homework.

"""

