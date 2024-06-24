import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('embedding_type_label_metric.csv')

# Display the first few rows of the dataframe
print(data.head())

# Check if 'embedding' needs transformation
def transform_embedding(embedding):
    # Assuming the embedding is stored as a string representation of a list
    if isinstance(embedding, str):
        return np.array(eval(embedding))
    else:
        return np.array(embedding)

# Apply the transformation to the 'embedding' column
data['embedding'] = data['embedding'].apply(transform_embedding)

# Stack embeddings into a 2D array if they are lists
embeddings = np.stack(data['embedding'].values)

# Encode 'template_type' using LabelEncoder
le_template_type = preprocessing.LabelEncoder()
data['template_type'] = le_template_type.fit_transform(data['template_type'])

# Combine embeddings with 'template_type' for the feature set
X = np.hstack((embeddings, data[['template_type']].values))

# Extract target variable
y = data['label']  # Assuming 'label' is the target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test, asset_id_train, asset_id_test = train_test_split(X, y, data['asset_id'], test_size=0.4, random_state=42)  # 70% training and 30% test

# Create model object
clf = MLPClassifier(hidden_layer_sizes=(6, 5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.001)

# Fit data onto the model
clf.fit(X_train, y_train)

# Predict the test set results
ypred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, ypred)
print(f'Accuracy: {accuracy}')

# Write predictions to file
with open('predictions.txt', 'w') as f:
    f.write('asset_id,predicted_label,actual_label\n')
    for asset_id, pred, actual in zip(asset_id_test, ypred, y_test):
        f.write(f'{asset_id},{pred},{actual}\n')
