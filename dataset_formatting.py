import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/nelgiriyewithana_emotions/text.csv", usecols=["text", "label"])

id_to_label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

X = data["text"]
y = data["label"]
y = y.map(id_to_label)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
val = pd.concat([X_val, y_val], axis=1)


train.to_csv("data/nelgiriyewithana_emotions/train.txt", sep=";", index=False, header=False)
test.to_csv("data/nelgiriyewithana_emotions/test.txt", sep=";", index=False, header=False)
val.to_csv("data/nelgiriyewithana_emotions/val.txt", sep=";", index=False, header=False)

print(f"Train length: {len(train)}")
print(f"Test length: {len(test)}")
print(f"Validation length: {len(val)}")