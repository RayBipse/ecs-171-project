import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
import random
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""**Get Data**"""

df = pd.read_csv("./mushroom/agaricus-lepiota.data")

"""**Generate Mapped Dataset**"""

mappings = {
    'cap-shape': {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'},
    'cap-surface': {'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'},
    'cap-color': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
    'bruises': {'t': 'bruises', 'f': 'no'},
    'odor': {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul', 'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy'},
    'gill-attachment': {'a': 'attached', 'd': 'descending', 'f': 'free', 'n': 'notched'},
    'gill-spacing': {'c': 'close', 'w': 'crowded', 'd': 'distant'},
    'gill-size': {'b': 'broad', 'n': 'narrow'},
    'gill-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray', 'r': 'green', 'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
    'stalk-shape': {'e': 'enlarging', 't': 'tapering'},
    'stalk-root': {'b': 'bulbous', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs', 'r': 'rooted', '?': 'missing'},
    'stalk-surface-above-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
    'stalk-surface-below-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
    'stalk-color-above-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
    'stalk-color-below-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
    'veil-type': {'p': 'partial', 'u': 'universal'},
    'veil-color': {'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'},
    'ring-number': {'n': 'none', 'o': 'one', 't': 'two'},
    'ring-type': {'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none', 'p': 'pendant', 's': 'sheathing', 'z': 'zone'},
    'spore-print-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'r': 'green', 'o': 'orange', 'u': 'purple', 'w': 'white', 'y': 'yellow'},
    'population': {'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 'v': 'several', 'y': 'solitary'},
    'habitat': {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'u': 'urban', 'w': 'waste', 'd': 'woods'}
}

df.columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',  'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',  'ring-type',  'spore-print-color', 'population',  'habitat']

#apply mappings to the dataframe
def map_features(df, mappings):
    for column in df.columns:
        if column in mappings:
            df[column] = df[column].map(mappings[column]).fillna(df[column])
    return df

# Apply the mappings to the dataframe
df_mapped = map_features(df, mappings)

"""**Check for and remove missing values**"""

for column in df_mapped.columns:
    if df_mapped[column].isnull().any():
        print(f"Column {column} has missing values")
        df_mapped = df_mapped[df_mapped[column].notnull()]

df = df_mapped

"""**DATA VISUALIZATION**

Frequency Bar Charts
"""

columns_to_plot = df.columns[:-1]

"""Generate Countplots to View Data correlations"""

categorical_cols = [col for col in df_mapped.columns if df_mapped[col].dtype == 'object']

"""**One-hot Encoding**"""

y = df_mapped['class'].map({'e': 1.0, 'p': 0.0})

# Drop the 'class' column
X = df_mapped.drop(['class'], axis=1)
# Initialize
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)
feature_names = encoder.get_feature_names_out(X.columns)
X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

y_df = pd.DataFrame(y).reset_index(drop=True)

df_encoded = pd.concat([y_df, X_encoded_df], axis=1)

df_encoded.head()

"""**Split dataset - in order to prevent data leakage**"""

X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)

"""**Helper function to assist with validation and testing**"""

def print_cv_results(X, y, model, params=dict()):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Split data into 10 folds
    kf = KFold(n_splits=10)
    accuracy, mse = np.zeros(kf.get_n_splits(X_train)), np.zeros(kf.get_n_splits(X_train))
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        train_index = slice(train_index[0], train_index[-1] + 1)
        test_index  = slice(test_index[0], test_index[-1] + 1)

        model.fit(X_train[train_index], y_train[train_index], **params)
        pred = model.predict(X_train[test_index]).round()
        accuracy[i] = accuracy_score(y_train[test_index], pred)
        mse[i] = mean_squared_error(y_train[test_index], pred)

    # Display the performance of each fold and the overall average
    performance_df = pd.DataFrame(np.transpose([np.arange(1, len(mse)+1), accuracy, mse]), columns=["Iteration", "Accuracy", "MSE"]).set_index("Iteration")
    print(f"Average accuracy = {np.average(accuracy)}")
    print(f"Average MSE = {np.average(mse)}")

"""**Logistic Regression**"""

print_cv_results(X_encoded_df, y, Pipeline([('pca', PCA(n_components=0.95)), ('logistic', LogisticRegression())]))

"""**Logistic Regression - Parameter Tuning**

- observing effects of changing the value of `max_iter`
"""

lg = LogisticRegression(max_iter=8000)
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

log_reg = LogisticRegression(max_iter=12000)

print_cv_results(X_train, y_train, log_reg)

from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=1, max_iter=300)

param_grid = {
    "max_iter": [1, 2, 3], # CHANGE
    "hidden_layer_sizes": [(14, 15, 12), (16, 13, 12), (13, 16, 14)],
    "activation": ['logistic', 'relu']
}

grid = GridSearchCV(estimator = mlp, param_grid = param_grid)
grid.fit(X_encoded_df, y)

# results of best model
best_model = mlp.set_params(**grid.best_params_)
print_cv_results(X_encoded_df, y, best_model)

"""**PCA + K-Fold Cross Validation**"""

X_train_cv, X_test, y_train_cv, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
accuracies = []  # Store accuracies for each fold

for train_index, val_index in kf.split(X_train_cv):
    X_train_fold, X_val_fold = X_train_cv.iloc[train_index], X_train_cv.iloc[val_index]
    y_train_fold, y_val_fold = y_train_cv.iloc[train_index], y_train_cv.iloc[val_index]

    # Fit PCA on the current training fold
    pca = PCA(n_components=0.95)  # Keep 95% of explained variance
    X_train_fold_pca = pca.fit_transform(X_train_fold)
    X_val_fold_pca = pca.transform(X_val_fold)

    # Train logistic regression
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train_fold_pca, y_train_fold)

    # Evaluation on the validation fold
    y_pred = log_reg.predict(X_val_fold_pca)
    accuracy = accuracy_score(y_val_fold, y_pred)
    accuracies.append(accuracy)

# Average accuracy across folds
avg_accuracy = np.mean(accuracies)
print(f"Average cross-validation accuracy: {avg_accuracy}")

"""**Kernel PCA + K-Fold Cross Validation**

**Naive Bayes**
"""

nb = CategoricalNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

from joblib import dump
dump(lg, './frontend/lg.joblib')
dump(mlp, './frontend/mlp.joblib')
dump(nb, './frontend/nb.joblib')
dump(encoder, './frontend/encoder.joblib')