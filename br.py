import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Process the data
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 3})
train = train.drop(['Name', 'Cabin', 'Ticket'], 1)

test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 3})
test = test.drop(['Name', 'Cabin', 'Ticket'], 1)

# Prep data for training
y = train['Survived'].fillna(-1)
x = train.drop('Survived', 1).fillna(-1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1001)

# Build and fit model
model = BayesianRidge()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

preditions = [round(val) for val in y_pred]

accuracy = accuracy_score(y_test, preditions)

print(f'Acc: {accuracy}')

# pred = model.predict(current)

# # Predict test data
# sub_pred = model.predict(test)

# sub_pred_r = [round(val) for val in sub_pred]

# # Create submission file
# sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': sub_pred_r})

# sub.to_csv('submission.csv', index=False)