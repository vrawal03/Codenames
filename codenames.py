import numpy as np
import re
from collections import Counter

# Tokenize corpus
def tokenize(corpus):
    # Split corpus into a list of words
    tokens = re.findall(r'\b\w+\b', corpus.lower())
    return tokens

# Build vocabulary
def build_vocabulary(tokens, vocab_size=None):
    # Count tokens and sort by frequency
    vocabulary = Counter(tokens)
    if vocab_size:
        # If a vocab size is specified, limit the vocabulary to the top 'vocab_size' terms
        vocabulary = {word: count for word, count in vocabulary.most_common(vocab_size)}
    return vocabulary

# Create word to index mapping
def word_to_one_hot(word, word_to_id, vocab_size):
    # Initialize a zero vector with the length of the vocabulary
    one_hot = np.zeros(vocab_size)
    # Set the index of the word to 1
    one_hot[word_to_id[word]] = 1
    return one_hot

# Generate training data for CBOW
def generate_training_data(tokens, word_to_id, window_size):
    N = len(tokens)
    X, Y = [], []

    for i in range(N):
        # Define the start and end index of the context
        start = max(0, i - window_size)
        end = min(N, i + window_size + 1)
        context = [tokens[j] for j in range(start, end) if j != i]
        # Use one-hot encoding for context words and target word
        context_vectors = [word_to_one_hot(w, word_to_id, len(word_to_id)) for w in context]
        target = word_to_one_hot(tokens[i], word_to_id, len(word_to_id))
        # Append the mean of the context vectors and the target word vector to the training data
        X.append(np.mean(context_vectors, axis=0))
        Y.append(target)
    return np.array(X), np.array(Y)

# Initialize network
def init_network(vocab_size, hidden_size):
    return {
        'W1': np.random.randn(vocab_size, hidden_size),
        'W2': np.random.randn(hidden_size, vocab_size)
    }

# Softmax activation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Forward pass
def forward_pass(x, model):
    h = np.dot(model['W1'].T, x)
    u = np.dot(model['W2'].T, h)
    y_c = softmax(u)
    return y_c, h, u

# Calculate error
def calculate_error(y, y_pred):
    return -np.sum(y * np.log(y_pred))

# Backpropagation
def backpropagate(x, y, y_pred, h, model, learning_rate):
    e = y_pred - y
    dW2 = np.outer(h, e)
    dW1 = np.outer(x, np.dot(model['W2'], e))
    model['W1'] -= learning_rate * dW1
    model['W2'] -= learning_rate * dW2
    return calculate_error(y, y_pred)

# Training loop
def train(X, Y, model, epochs, learning_rate):
    for epoch in range(epochs):
        loss = 0
        for x, y in zip(X, Y):
            y_pred, h, u = forward_pass(x, model)
            loss += backpropagate(x, y, y_pred, h, model, learning_rate)
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Find the closest word
def find_closest_word(embedding, word_to_id, model):
    min_dist = float('inf')
    closest_word = None
    for word, index in word_to_id.items():
        word_embedding = model['W1'][index]
        distance = np.linalg.norm(embedding - word_embedding)
        if (distance < min_dist) and (distance > 0):
            min_dist = distance
            closest_word = word
    return closest_word

# Main execution
corpus = '''I love ice cream. Ice cream is cold and delicious. Cold ice and sweet cream. Ice cream is the best. '''

tokens = tokenize(corpus)
vocabulary = build_vocabulary(tokens)
word_to_id = {word: i for i, word in enumerate(vocabulary.keys())}
id_to_word = {i: word for i, word in enumerate(vocabulary.keys())}

window_size = 2
X, Y = generate_training_data(tokens, word_to_id, window_size)

hidden_size = 100  # Size of the hidden layer
model = init_network(len(word_to_id), hidden_size)

epochs = 1000
learning_rate = 0.001
train(X, Y, model, epochs, learning_rate)

# Example of finding the closest word
word_embedding = model['W1'][word_to_id['cream']]  # Replace 'example' with your word
closest_word = find_closest_word(word_embedding, word_to_id, model)
print(f'The closest word to "cream" is "{closest_word}".')