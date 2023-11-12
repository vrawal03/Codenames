import numpy as np
import re
from collections import Counter

# Set the seed for random number generation
np.random.seed(0)

# Function to extract words from a given string
def extract_words(input_text):
    word_pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return word_pattern.findall(input_text.lower())

# Sample text for processing
sample_text = '''Ice cream is cold. Ice cream is sweet. Cold ice and sweet cream. Ice cream is the best.'''

# Extracting words from the text
word_list = extract_words(sample_text)

# Function to create mappings between words and their numerical representations
def create_mappings(words):
    word_index_map = {}
    index_word_map = {}
    unique_words = set(words)

    for index, word in enumerate(unique_words):
        word_index_map[word] = index
        index_word_map[index] = word

    return word_index_map, index_word_map

# Creating mappings
index_map, word_map = create_mappings(word_list)

# Function to generate training pairs
def generate_pairs(words, mapping, window_size):
    input_data = []
    target_data = []
    num_words = len(words)

    for index in range(num_words):
        for i in range(max(0, index - window_size), index):
            input_data.append(vector_encode(mapping[words[index]], len(mapping)))
            target_data.append(vector_encode(mapping[words[i]], len(mapping)))

        for j in range(index, min(num_words, index + window_size + 1)):
            if index == j:
                continue
            input_data.append(vector_encode(mapping[words[index]], len(mapping)))
            target_data.append(vector_encode(mapping[words[j]], len(mapping)))

    return np.array(input_data), np.array(target_data)

# Function for one-hot encoding
def vector_encode(index, size):
    vector = [0] * size
    vector[index] = 1
    return vector

# Generating training data
input_vectors, target_vectors = generate_pairs(word_list, index_map, 2)

# Initializing the neural network
def initialize_network(vocab_size, embedding_dim):
    return {
        "layer1_weights": np.random.randn(vocab_size, embedding_dim),
        "layer2_weights": np.random.randn(embedding_dim, vocab_size)
    }

# Creating a model
neural_model = initialize_network(len(index_map), 10)

# Forward propagation function
def propagate_forward(network, input_vec, return_intermediate=False):
    intermediate_values = {}
    layer1_output = np.dot(input_vec, network["layer1_weights"])
    layer2_output = np.dot(layer1_output, network["layer2_weights"])
    softmax_output = apply_softmax(layer2_output)

    intermediate_values["layer1_output"] = layer1_output
    intermediate_values["layer2_output"] = layer2_output
    intermediate_values["softmax_output"] = softmax_output

    return intermediate_values

# Softmax function
def apply_softmax(input_vec):
    softmax_result = []
    for vec in input_vec:
        exp_vec = np.exp(vec)
        softmax_result.append(exp_vec / np.sum(exp_vec))
    return softmax_result

# Backward propagation function
def propagate_backward(network, input_vec, target_vec, learning_rate):
    intermediate_values = propagate_forward(network, input_vec, return_intermediate=True)
    error_layer2 = intermediate_values["softmax_output"] - target_vec
    gradient_layer2 = np.dot(intermediate_values["layer1_output"].T, error_layer2)
    error_layer1 = np.dot(error_layer2, network["layer2_weights"].T)
    gradient_layer1 = np.dot(input_vec.T, error_layer1)

    network["layer1_weights"] -= learning_rate * gradient_layer1
    network["layer2_weights"] -= learning_rate * gradient_layer2

    return calculate_loss(intermediate_values["softmax_output"], target_vec)

# Cross-entropy loss function
def calculate_loss(predicted, actual):
    return -np.sum(np.log(predicted) * actual)

# Training the model
num_iterations = 1000
alpha = 0.001
loss_history = []
for i in range(num_iterations):
    loss = propagate_backward(neural_model, input_vectors, target_vectors, alpha)
    loss_history.append(loss)

# One-hot encoding for a specific word
test_word_vector = vector_encode(index_map["cream"], len(index_map))

# Reshaping and feeding into the model
test_word_vector_reshaped = np.array([test_word_vector])
output_cache = propagate_forward(neural_model, test_word_vector_reshaped, return_intermediate=True)
output_result = output_cache['softmax_output'][0]

# Displaying the words based on the softmax output
sorted_indices = np.argsort(output_result)[::-1]
sorted_words = [word_map[id] for id in sorted_indices]
print("Sorted words by closeness to cream: ") 
for word in sorted_words:
  print(word)

# sorted_words[0] is 'ice', as expected.

# Extract word embeddings for K-means clustering
word_embeddings = neural_model["layer1_weights"]
word_embeddings_np = np.array(word_embeddings)

k = 3
kmeans = KMeans(n_clusters=k, random_state=0)

kmeans.fit(word_embeddings_np)

labels = kmeans.labels_
print("Cluster assignments:", labels)
# Check if the KMeans model has been fitted correctly
if kmeans.labels_ is not None:
  # Initialize an empty dictionary for the word to cluster map
  word_cluster_map = {}

  # Check if index_map is initialized and not empty
  if index_map is not None:
    # Iterate through each word and its index in the index_map
    for word, index in index_map.items():
      # Get the cluster assignment for each word and add it to the dictionary
      word_cluster_map[word] = kmeans.labels_[index]

  print(word_cluster_map)

else:
  print("Error: KMeans model not fitted properly.")
