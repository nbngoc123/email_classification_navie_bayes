import numpy as np

def create_vocab(messages):
    all_words = []
    for message in messages:
        if message not in all_words:
            all_words.append(message)
    return all_words

# def create_feature(messages, vocab):
#     feature_vector = np.zeros(len(vocab))
#     for message in messages:
#         if message in vocab:
#             feature_vector[vocab.index(message)] += 1
#     return feature_vector

def create_feature_vector(message_tokens, vocab):
    """
    Chuyển đổi một tin nhắn (danh sách các token) thành một vector Bag of Words.
    """
    vocab_map = {word: i for i, word in enumerate(vocab)}
    print(vocab_map)
    feature_vector = np.zeros(len(vocab))
    for token in message_tokens:
        if token in vocab_map:
            print(vocab_map[token])
            feature_vector[vocab_map[token]] += 1
    return feature_vector

# test
# Từ vựng (vocab) gồm 5 từ
vocab = ["hello", "world", "goodbye", "machine", "learning"]

# Một tin nhắn (danh sách token)
message = ["hello", "world", "machine", "learning", "hello", "hello"]

# output1 = create_feature(message, vocab)
# print("Output from create_feature:", output1)

output2 = create_feature_vector(message, vocab)
print("Output from create_feature_vector:", output2)