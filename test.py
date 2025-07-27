from load_data import DataLoader
from training import Training
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = DataLoader('data/2cls_spam_text_cls.csv')
messages, y = data.load_data()

VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

X_train, X_val, y_train, y_val = train_test_split(
    messages, y,
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

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

trainer = Training()
trainer.fit(X_train, y_train)

y_val_pred = trainer.predict(X_val)
y_test_pred = trainer.predict(X_test)

print(f'Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}')
print(f'Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}')