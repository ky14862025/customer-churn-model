# -*- coding: utf-8 -*-
"""Customer_churn_Assignment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xcVlIedmrVI1GoMZIM-auuNDPaWdqc83
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle as pkl
import seaborn as sns

customer_churn_df= pd.read_csv('/content/drive/MyDrive/CustomerChurn_dataset.csv')

customer_churn_df

columns_to_drop = ['customerID']

customer_churn_df = customer_churn_df.drop(columns_to_drop, axis=1)

customer_churn_df.head()

customer_churn_df.isnull().sum()

customer_churn_df.info()

customer_churn_df['TotalCharges'] = pd.to_numeric(customer_churn_df['TotalCharges'], errors='coerce', downcast='float')



customer_churn_df.info()

customer_churn_df.isnull().sum()

imputer = SimpleImputer(strategy = 'mean')
customer_churn_df['TotalCharges'] = customer_churn_df['TotalCharges'].replace('', np.NaN)

Total_charges = customer_churn_df['TotalCharges'].values.reshape(-1,1)

customer_churn_df['TotalCharges'] = imputer.fit_transform(Total_charges)

customer_churn_df['Contract'].value_counts()

customer_churn_df.isnull().sum()







customer_churn_df.info()

encoder = LabelEncoder()
encoded_df = customer_churn_df.copy()
for column in encoded_df.columns:
  if encoded_df[column].dtype == 'object':
    encoded_df[column] =  encoder.fit_transform(encoded_df[column])

encoded_df
y= encoded_df['Churn']
encoded_df= encoded_df.drop('Churn',axis=1)

sc= StandardScaler()
scaled_df = sc.fit_transform(encoded_df)

churn_df = pd.DataFrame(scaled_df,columns=encoded_df.columns)
data = pd.concat([churn_df, y], axis=1)

data

"""Feature selcection"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# Split the data into features (X) and target variable (y)
X = data.drop('Churn', axis=1)
y = data['Churn']



correlations = X.corrwith(pd.Series(y)).abs().sort_values(ascending=False)


k = 7
top_features = correlations.head(k).index
selected_features = X[top_features]


cv = StratifiedKFold(7)
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(estimator=model, step=1, cv=cv)
rfecv.fit(selected_features, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print("Selected features: %s" % ', '.join(selected_features.columns[rfecv.support_]))
selected_features

selected_features = ['Contract', 'tenure', 'OnlineSecurity', 'TechSupport', 'TotalCharges', 'OnlineBackup', 'MonthlyCharges']

important =customer_churn_df[['Contract', 'tenure', 'OnlineSecurity', 'TechSupport', 'TotalCharges', 'OnlineBackup', 'MonthlyCharges']]

important.info()

numerical_features = ['tenure', 'TotalCharges', 'MonthlyCharges']
categorical_features =['Contract', 'OnlineSecurity', 'TechSupport', 'OnlineBackup']

"""Exploratory Data Analysis"""

for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Churn', y=feature, data=customer_churn_df)
    plt.title(f'{feature} vs. Churn')
    plt.show()




    for feature in categorical_features:
      plt.figure(figsize=(8, 6))
      sns.countplot(x=feature, hue='Churn', data=customer_churn_df)
      plt.title(f'{feature} vs. Churn')
      plt.show()



X = encoded_df[selected_features]
y = data['Churn']

# Use StandardScaler to scale your features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Create a DataFrame from the scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Now, X_scaled_df should contain your scaled features

X_scaled_df

"""Training with MLP and Keras Tuner"""

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# Keras Functional API model
input_layer = Input(shape=(X_train.shape[1],))
hidden_layer_1 = Dense(32, activation='relu')(input_layer)
hidden_layer_2 = Dense(24, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(12, activation='relu')(hidden_layer_2)
output_layer = Dense(1, activation='sigmoid')(hidden_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

_, accuracy = model.evaluate(X_train, y_train)
accuracy*100
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.4f}')

!pip install keras-tuner

import keras_tuner
from tensorflow import keras

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(X_train.shape[1],)))

    # Tune the number of hidden layers and units
    for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=4)):
        model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=96, step=32),
                             activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh'])))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Tune the learning rate
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    return model

build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.Hyperband(
  hypermodel=build_model,
  objective='val_accuracy',
  max_epochs=100,
  factor=3,
  directory='tuning_dir',
  project_name='samples')

tuner.search(X_train, y_train, epochs=30 ,validation_data=(X_test, y_test))

tuner.search_space_summary()

tuner.results_summary()

model = tf.keras.Model(...)
checkpoint = tf.train.Checkpoint(model)

# Save a checkpoint to /tmp/training_checkpoints-{save_counter}. Every time
# checkpoint.save is called, the save counter is increased.
save_path = checkpoint.save('/tmp/training_checkpoints')

# Restore the checkpointed values to the `model` object.
checkpoint.restore(save_path)

best_model = tuner.get_best_models(num_models=2)[0]

best_model.summary()

"""Evaluation And Accuracy Testing"""

test_accuracy = best_model.evaluate(X_test, y_test)[1]
print(f"Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = best_model.predict(X_test)
y_pred_binary = (y_pred>0.5).astype(int)


# Calculate AUC
accuracy = accuracy_score(y_test,y_pred_binary )
auc_score = roc_auc_score(y_test,y_pred)
print("Initial Model Accuracy:", accuracy)
print("Initial Model Auc Score:", auc_score)



best_model.save('churn_assign.h5')



with open('scaler.pkl','wb') as file:
  pkl.dump(sc,file)

with open('label_encoder.pkl', 'wb') as file:
    pkl.dump(encoder, file)