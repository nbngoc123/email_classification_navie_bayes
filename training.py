import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def create_vocab(messages):
    all_words = []
    for tokens in messages:
        for token in tokens:
            if token not in all_words:
                all_words.append(token)
    return all_words

def create_feature_vector(message_tokens, vocab):
    """
    Chuyển đổi một tin nhắn (danh sách các token) thành một vector Bag of Words.
    """

    vocab_map = {word: i for i, word in enumerate(vocab)}
    feature_vector = np.zeros(len(vocab))
    for token in message_tokens:
        if token in vocab_map:
            feature_vector[vocab_map[token]] += 1
    return feature_vector

class Training:
    def __init__(self, messages, labels):
        self.messages = messages
        self.labels = labels
        self.vocab = create_vocab(messages)
        self.X = np.array([create_feature_vector(message, self.vocab) for message in messages])

    def train(self):
        model = MultinomialNB()
        print('Start training...')
        model.fit(self.X, self.labels)
        print('Training completed!')
        return self.X, model 
    
    def predict(self, model, x_test, y_test, x_val, y_val):
        y_test_pred = model.predict(x_test)
        y_val_pred = model.predict(x_val)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f'Val accuracy: {val_accuracy}')
        print(f'Test accuracy: {test_accuracy}')