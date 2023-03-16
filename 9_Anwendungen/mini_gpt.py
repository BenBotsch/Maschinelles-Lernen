#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 20:22:19 2023

@author: ben
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import re
import string
import random


"""
## Implement a Transformer block as a layer
"""


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    The causal_attention_mask-function takes 
    four parameters batch_size, n_dest, 
    n_src, and dtype and returns a tensor 
    representing a mask to be used in causal 
    attention.
    
    Parameters:
    -----------
    batch_size : int
        The number of sequences in a batch.
    n_dest : int
        The number of positions in the output sequence.
    n_src : int
        The number of positions in the input sequence.
    dtype :
        The data type of the returned tensor.
    
    Returns:
    --------
    A tensor of shape (batch_size, n_dest, 
    n_src) representing a mask to be used 
    in causal attention.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    """
    The TransformerBlock class represents 
    a single block of a transformer neural 
    network. It takes four parameters 
    embed_dim, num_heads, ff_dim, and rate, 
    and applies multi-head attention and a 
    feedforward neural network (FFN) to 
    the input.
    
    Parameters:
    -----------
    embed_dim : int
        The dimensionality of the embedding 
        space.
    num_heads : int
        The number of attention heads.
    ff_dim : int
        The dimensionality of the intermediate 
        layer in the FFN.
    rate : float
        The dropout rate to use.
    
    Returns:
    --------
    A tensor of the same shape as the input 
    tensor representing the output of the 
    transformer block.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        """
        Passes the inputs through the transformer 
        block and returns the output tensor.
        """
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement an embedding layer
Create two seperate embedding layers: one for tokens and one for token index
(positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    """
    A layer that combines token embeddings 
    and positional embeddings.
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        """
        Initialize the layer with the 
        given parameters.

        Parameters:
        -----------
            maxlen : int
                The maximum sequence length.
            vocab_size : int
                The size of the token vocabulary.
            embed_dim : int
                The size of the embedding vectors.
        """
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        """
        Compute the output of the layer 
        given an input tensor x.

        Parameters:
        -----------
            x : Tensor
                The input tensor, with shape 
                (batch_size, seq_length).

        Returns:
        --------
            Tensor : 
                The output tensor
        """
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


"""
## Implement the miniature GPT model
"""
vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer


def create_model():
    """
    Create a transformer model for 
    sequence-to-sequence tasks.

    The model takes a sequence of token 
    indices as input and predicts the next 
    token in the sequence. It consists of 
    a TokenAndPositionEmbedding layer 
    followed by a stack of TransformerBlock 
    layers, and a final Dense layer that
    outputs the logits for each token
    in the vocabulary.

    Returns:
    --------
        Model: 
            A Keras model that takes an input 
            tensor with shape (batch_size, 
            seq_length) and outputs a tuple 
            of two tensors: the logits for 
            each token in the vocabulary and 
            the intermediate tensor produced by 
            the last TransformerBlock layer.
    """
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model


"""
## Prepare the data for word-level language modelling
Download the IMDB dataset and combine training and validation sets for a text
generation task.
"""

"""shell
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
"""


batch_size = 128

# The dataset contains each review in a separate text file
# The text files are present in four different folders
# Create a list all files
filenames = []
directories = [
    "aclImdb/train/pos",
    "aclImdb/train/neg",
    "aclImdb/test/pos",
    "aclImdb/test/neg",
]
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

print(f"{len(filenames)} files")

# Create a dataset from text files
random.shuffle(filenames)
text_ds = tf.data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


# Create a vectorization layer and adapt it to the text
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices


def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels, num_parallel_calls=tf.data.AUTOTUNE)
text_ds = text_ds.prefetch(tf.data.AUTOTUNE)


"""
## Implement a Keras callback for generating text
"""


class TextGenerator(keras.callbacks.Callback):
    """
    Callback for generating text at the end 
    of each epoch.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1):
        """
        Initializes a new instance of the 
        TextGenerator class.
        
        Parameters:
        -----------
            max_tokens : int
                The maximum number of tokens 
                to generate.
            start_tokens : list
                The list of start tokens to 
                generate text from.
            index_to_word : list
                A list of strings.
            top_k : int
                The number of top logits to consider 
                when sampling from the output
                distribution. Defaults to 10.
            print_every : int
                The number of epochs between text 
                generation. Defaults to 1.
        """
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        """
        Samples a token index from the 
        output logits.

        Parameters:
        -----------
            logits : np.ndarray
                The output logits of the model.

        Returns:
        --------
            int : 
                A token index sampled from 
                the output distribution.
        """
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        """
        Converts a token index to its 
        corresponding word string.

        Parameters:
        -----------
            number : int
                A token index.

        Returns:
        --------
            str : 
                The word that corresponds 
                to the given token index.
        """
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        """
        Generates text and prints it 
        to the console.

        Parameters:
        -----------
            epoch : int
                The current epoch number.
            logs : dict
                A dictionary of metrics for 
                the current epoch.
        """
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")


# Tokenize starting prompt
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

start_prompt = "this movie is"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)


    
"""
## Train the model
Note: This code should preferably be run on GPU.
"""

model = create_model()

model.fit(text_ds, verbose=2, epochs=3, callbacks=[text_gen_callback])
    
    
    
    
    
    