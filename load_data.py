import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocess_text import text_preprocessing

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        df = pd.read_csv(self.path)[:10]
        print('load data from csv file successfully!')

        messages = df['Message'].values.tolist()
        labels = df['Category'].values.tolist()
        
        le = LabelEncoder()
        y = le.fit_transform(labels)
        print(f'Classes: {le.classes_}')
        print(f'Encoded labels: {y}')

        messages = [
            text_preprocessing(message) for message in messages
        ]
        print('text preprocessing successfully!')
        print('ah... em đã sẵn sàng darlling chiếm lấy em đi ah....ah...')
        return messages, y
# test  
# data = DataLoader('data/2cls_spam_text_cls.csv')
# data.load_data()