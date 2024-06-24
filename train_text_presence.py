# import torch.cuda
# from sklearn.model_selection import train_test_split
# import torch.nn as nn
# import pandas as pd
# import numpy as np

# from text_presence import TextPresenceDataset, CLASS_TO_INDEX
# from text_presence_models import TextPresenceSLP, TextPresenceMLP
# from train import BaseTrainer, DEVICE

# torch.cuda.empty_cache()
# torch.manual_seed(42)

# # data = pd.read_csv("data/text_presence/v2/a2005128_a2007426.csv")
# data=pd.read_csv("labeled_data_metric.csv")
# print(len(data))
# train_valid, test_data = train_test_split(data, test_size=0.05, random_state=35)
# train_data, valid_data = train_test_split(train_valid, test_size=0.1, random_state=35)

# print("train:", train_data.shape)
# print("valid:", valid_data.shape)
# print("test:", test_data.shape)

# train_dataset = TextPresenceDataset(train_data.index.values)
# val_dataset = TextPresenceDataset(valid_data.index.values)
# test_dataset = TextPresenceDataset(test_data.index.values)

# batch_size = 128
# epochs = 2000
# patience = 25

# print("batch_size:", batch_size)
# print("epochs:", epochs)
# print("patience:", patience)

# net = TextPresenceMLP()
# criterion = nn.CrossEntropyLoss()

# if torch.cuda.device_count() > 1 and DEVICE != "cpu":
#     net = torch.nn.DataParallel(net)

# print(net)
# net.to(DEVICE)
# criterion.to(DEVICE)

# trainer = BaseTrainer(
#     train_dataset=train_dataset,
#     val_dataset=val_dataset,
#     test_dataset=test_dataset,
#     model=net,
#     criterion=criterion,
#     batch_size=batch_size,
#     epochs=epochs,
#     patience=patience,
#     labels=np.array(list(CLASS_TO_INDEX.values())),
#     target_names=np.array(list(CLASS_TO_INDEX.keys())),
# )
# model_arch = "mlp"
# model_name = "text_presence"
# version = "v3"
# embeddings_type = "adobeone-h14"
# save_dir = f"artifacts/{model_name}/{version}/{model_arch}"
# print(save_dir)
# print(model_arch)
# weights_path = f"{save_dir}/best_checkpoint_{model_name}_multiclass_{embeddings_type}_{len(data)}.pth"
# pred_classes_save_path = f"{save_dir}/pred_classes_{embeddings_type}_{len(data)}.npy"
# true_classes_save_path = f"{save_dir}/true_classes_{len(data)}.npy"
# trainer.main(weights_path, pred_classes_save_path, true_classes_save_path)

# print(weights_path)
# print(pred_classes_save_path)
# print(true_classes_save_path)

# import numpy as np
# import pandas as pd

# # Load data
# data=pd.read_csv('embedding_type_label_metric.csv')

# data.head()

# from sklearn import preprocessing

# # Creating labelEncoder
# le = preprocessing.LabelEncoder()

# # Converting string labels into numbers.
# data['salary']=le.fit_transform(data['salary'])
# data['Departments ']=le.fit_transform(data['Departments '])

# X=data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Departments ', 'salary']]
# y=data['left']

# # Import train_test_split function
# from sklearn.model_selection import train_test_split

# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test

# from sklearn.neural_network import MLPClassifier

# # Create model object
# clf = MLPClassifier(hidden_layer_sizes=(6,5),
#                     random_state=5,
#                     verbose=True,
#                     learning_rate_init=0.01)

# # Fit data onto the model
# clf.fit(X_train,y_train)

# ypred=clf.predict(X_test)

# # Import accuracy score 
# from sklearn.metrics import accuracy_score

# # Calcuate accuracy
# accuracy_score(y_test,ypred)

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
