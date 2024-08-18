import numpy as np

def softMax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def SelfAttention(x):
    batch_size, seq_len, d_model = x.shape
    
    d_k = d_v = d_model
    Qw = np.random.rand(d_model, d_k)
    Kw = np.random.rand(d_model, d_k)
    Vw = np.random.rand(d_model, d_v)
    
    Q = np.dot(x, Qw)
    K = np.dot(x, Kw)
    V = np.dot(x, Vw)
    
    AttentionScores = np.matmul(Q, K.transpose(0, 2, 1))
    AttentionScores /= np.sqrt(d_k)
    
    AttentionWeight = softMax(AttentionScores)
    
    output = np.matmul(AttentionWeight, V)
    return output

def fnn(x, d_ff):
    batch_size, seq_len, d_model = x.shape
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.zeros(d_model)
    
    x = np.dot(x, W1) + b1
    x = np.maximum(0, x)
    x = np.dot(x, W2) + b2
    return x

def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + epsilon)
    return x

def encoder(x, d_ff):
    attn_output = SelfAttention(x)
    x = x + attn_output
    x = layer_norm(x)
    ff_output = fnn(x, d_ff)
    x = x + ff_output
    x = layer_norm(x)
    return x

def decoder(x, enc_output, d_ff):
    self_attn_output = SelfAttention(x)
    x = x + self_attn_output
    x = layer_norm(x)
    
    cross_attn_output = SelfAttention(x)  # In practice, should use enc_output, simplified here
    x = x + cross_attn_output
    x = layer_norm(x)
    
    ff_output = fnn(x, d_ff)
    x = x + ff_output
    x = layer_norm(x)
    return x

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angles = pos / np.power(10000, (2 * (i // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles[np.newaxis, :, :]

def embedding_layer(input_ids, d_model, vocab_size):
    embeddings = np.random.randn(vocab_size, d_model)
    return embeddings[input_ids]

def classification_head(x, num_classes):
    batch_size, seq_len, d_model = x.shape
    W = np.random.randn(d_model, num_classes)
    b = np.zeros(num_classes)
    
    x = x.mean(axis=1)
    x = np.dot(x, W) + b
    return softMax(x)

def tokenize_and_encode(texts, vocab):
    tokenized = [[vocab.get(word, 0) for word in text.split()] for text in texts]
    max_len = max(len(t) for t in tokenized)
    padded = [t + [0] * (max_len - len(t)) for t in tokenized]
    return np.array(padded)

def print_classification(output):
    labels = ["Negative", "Positive"]
    for i, probs in enumerate(output):
        predicted_class = np.argmax(probs)
        print(f"Statement {i + 1} is classified as: {labels[predicted_class]}")

texts = ["I love myself", "I am going to hate myself"]
vocab = {"I": 1, "love": 2, "myself": 3, "I": 4, "am": 5, "going": 6, "to": 7, "hate": 8, "myself": 9}
vocab_size = 10

batch_size = len(texts)
seq_len = max(len(text.split()) for text in texts)
d_model = 16
d_ff = 32
num_classes = 2

input_ids = tokenize_and_encode(texts, vocab)
x = embedding_layer(input_ids, d_model, vocab_size)
x += positional_encoding(seq_len, d_model)

encoder_output = encoder(x, d_ff)
decoder_output = decoder(x, encoder_output, d_ff)

classification_output = classification_head(decoder_output, num_classes)

print("Classification Output Shape:", classification_output.shape)
print("Classification Output:", classification_output)

print_classification(classification_output)
