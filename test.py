from load_data import DataLoader
from training import Training
from sklearn.model_selection import train_test_split

data = DataLoader('data/2cls_spam_text_cls.csv')
messages, y = data.load_data()
train = Training(messages, y)
X , model = train.train()

VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VAL_SIZE,
    shuffle=True,
    random_state=SEED
)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
    test_size=TEST_SIZE,
    shuffle=True,
    random_state=SEED
)
train.predict(model, X_test, y_test, X_val, y_val)