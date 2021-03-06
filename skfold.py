import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

# Import the data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Process the data
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 3})

df = pd.DataFrame(train['Name'].str.split(',').tolist())
df2 = pd.DataFrame(df[1].str.split('.').tolist())
train['Title'] = df2[0]

labels, levels = pd.factorize(train['Title'])

titles = pd.DataFrame(labels)

train['Title'] = titles
train = train.drop(['Name', 'Cabin', 'Ticket'], 1)


test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 3})

tdf = pd.DataFrame(test['Name'].str.split(',').tolist())
tdf2 = pd.DataFrame(tdf[1].str.split('.').tolist())
test['Title'] = tdf2[0]

tlabels, levels = pd.factorize(test['Title'])

ttitles = pd.DataFrame(tlabels)

test['Title'] = ttitles
test = test.drop(['Name', 'Cabin', 'Ticket'], 1)

# print(train.head())

# Prep data for training
y = train['Survived']
x = train.drop('Survived', 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1001)

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

# Stratified K Fold
n_splits = 5
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15) 

results = cross_val_score(model, x, y, cv=folds)

print(f'Acc: {results.mean()*100}% ({results.std()*100}%)')

# Fit the data
model.fit(x, y)

# Predict test data
sub_pred = model.predict(test)

sub_pred_r = [round(val) for val in sub_pred]

# Create submission file
sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': sub_pred_r})

sub.to_csv('skfold_submission.csv', index=False)