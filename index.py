import pandas as pd
from xgboost import XGBClassifier
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
y = train['Survived']
x = train.drop('Survived', 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.75, random_state=1001)

# Prams for classifier
n_estimators = 90
learning_rate = .03
max_depth = 4
subsample = 1
colsample_bytree = 1
gamma = 0
max_delta_step = 0
min_child_weight = 1

# Build and fit model
model = XGBClassifier(n_estimators = n_estimators,
                      max_depth = max_depth,
                      learning_rate = learning_rate,
                      subsample = subsample,
                      colsample_bytree = colsample_bytree,
                      gamma = gamma,
                      max_delta_step = max_delta_step,
                      min_child_weight = min_child_weight
                      )
model.fit(X_train, y_train)

# Get train predicitons
y_pred = model.predict(X_test)
preditions = [round(val) for val in y_pred]

accuracy = accuracy_score(y_test, preditions)

print(f'Acc: {accuracy*100}%')

# Predict test data
sub_pred = model.predict(test)

sub_pred_r = [round(val) for val in sub_pred]

# Create submission file
sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': sub_pred_r})

sub.to_csv('submission.csv', index=False)