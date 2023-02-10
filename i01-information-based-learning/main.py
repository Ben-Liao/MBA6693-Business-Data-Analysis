# import pandas as pd
# import numpy as np
# from collections import Counter
# from math import log2

# def calc_entropy(p):
#     return -p * log2(p) - (1 - p) * log2(1 - p)

# def calc_gini(p):
#     return 1 - p ** 2 - (1 - p) ** 2

# def get_data_entropy(data, label_col='PlayTennis'):
#     labels = data[label_col].values
#     prob = Counter(labels)[1] / len(labels)
#     return calc_entropy(prob)

# def get_data_gini(data, label_col='PlayTennis'):
#     labels = data[label_col].values
#     prob = Counter(labels)[1] / len(labels)
#     return calc_gini(prob)

# def get_split_entropy(data, attribute, label_col='PlayTennis'):
#     total_entropy = 0
#     attr_values = data[attribute].unique()
#     for attr_value in attr_values:
#         sub_data = data[data[attribute] == attr_value]
#         prob = len(sub_data) / len(data)
#         total_entropy += prob * get_data_entropy(sub_data, label_col)
#     return total_entropy

# def get_split_gini(data, attribute, label_col='PlayTennis'):
#     total_gini = 0
#     attr_values = data[attribute].unique()
#     for attr_value in attr_values:
#         sub_data = data[data[attribute] == attr_value]
#         prob = len(sub_data) / len(data)
#         total_gini += prob * get_data_gini(sub_data, label_col)
#     return total_gini

# def find_best_split_entropy(data, attributes, label_col='PlayTennis'):
#     best_entropy = float('inf')
#     best_attribute = None
#     for attribute in attributes:
#         entropy = get_split_entropy(data, attribute, label_col)
#         if entropy < best_entropy:
#             best_entropy = entropy
#             best_attribute = attribute
#     return best_attribute

# def find_best_split_gini(data, attributes, label_col='PlayTennis'):
#     best_gini = float('inf')
#     best_attribute = None
#     for attribute in attributes:
#         gini = get_split_gini(data, attribute, label_col)
#         if gini < best_gini:
#             best_gini = gini
#             best_attribute = attribute
#     return best_attribute

# def id3(data, attributes, label_col='PlayTennis', metric='entropy'):
#     if len(attributes) == 0:
#         return Counter(data[label_col]).most_common(1)[0][0]
#     if len(data[label_col].unique()) == 1:
#         return data[label_col].iloc


import pandas as pd

# Read in data from the URL
url = "https://raw.githubusercontent.com/Ben-Liao/MBA6693-Business-Data-Analysis/main/i01-information-based-learning/data/tennis.txt"
df = pd.read_csv(url, delimiter="\t")

# Convert the data to a numerical representation
data = pd.get_dummies(df, columns=["Outlook", "Temperature", "Humidity", "Wind"], prefix=["Outlook", "Temperature", "Humidity", "Wind"])

# Define the features and target variable
X = data.drop("PlayTennis", axis=1)
y = data["PlayTennis"]

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the ID3 decision tree model using entropy as the impurity measure
from sklearn.tree import DecisionTreeClassifier
model_entropy = DecisionTreeClassifier(criterion="entropy", random_state=0)
model_entropy.fit(X_train, y_train)

# Train the ID3 decision tree model using Gini index as the impurity measure
model_gini = DecisionTreeClassifier(criterion="gini", random_state=0)
model_gini.fit(X_train, y_train)

# Evaluate the performance of the models using accuracy
from sklearn.metrics import accuracy_score
y_pred_entropy = model_entropy.predict(X_test)
y_pred_gini = model_gini.predict(X_test)

print("Accuracy using entropy:", accuracy_score(y_test, y_pred_entropy))
print("Accuracy using Gini index:", accuracy_score(y_test, y_pred_gini))
