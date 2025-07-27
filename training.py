import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class Training:
    def __init__(self, model=MultinomialNB()):
        """
        Khởi tạo class Training.
        - model: Mô hình scikit-learn sẽ được sử dụng.
        - vocab: Sẽ được tạo khi gọi phương thức fit.
        - vocab_map: Dictionary để tra cứu chỉ số từ nhanh chóng.
        """
        self.model = model
        self.vocab = None
        self.vocab_map = None
    
    def _create_vocab(self, messages):
        """
        Tạo bộ từ vựng từ danh sách các tin nhắn đã được token hóa.
        Sử dụng set để đạt hiệu năng cao.
        """
        # Làm phẳng danh sách và sử dụng set để lấy các từ duy nhất
        all_words = {word for message in messages for word in message}
        # Sắp xếp để đảm bảo thứ tự nhất quán
        self.vocab = sorted(list(all_words))
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}

    def _create_feature_vector(self, message_tokens):
        """Chuyển đổi một tin nhắn thành vector Bag of Words."""
        feature_vector = np.zeros(len(self.vocab))
        for token in message_tokens:
            if token in self.vocab_map:
                feature_vector[self.vocab_map[token]] += 1
        return feature_vector

    def fit(self, messages, labels):
        """
        Huấn luyện mô hình từ dữ liệu thô (đã được tiền xử lý).
        Bao gồm việc tạo từ vựng, vector hóa và huấn luyện mô hình.
        """
        print('Creating vocabulary and feature vectors...')
        self._create_vocab(messages)
        X_train = np.array([self._create_feature_vector(msg) for msg in messages])
        
        print('Start training...')
        self.model.fit(X_train, labels)
        print('Training completed!')
    
    def predict(self, messages):
        """Dự đoán nhãn cho các tin nhắn mới."""
        X = np.array([self._create_feature_vector(msg) for msg in messages])
        return self.model.predict(X)