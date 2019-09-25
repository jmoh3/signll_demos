import torch
import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

df = pd.read_csv('yelp_reviews.csv')

df = df.drop('Unnamed: 0', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['pos_neg'].values, test_size=0.2)

vectorizer = CountVectorizer(lowercase=True)

X_train_matrix = vectorizer.fit_transform(X_train)
X_test_matrix = vectorizer.transform(X_test)

X_train_tensor = torch.tensor(X_train_matrix.toarray()).float()
X_test_tensor = torch.tensor(X_test_matrix.toarray()).float()

w = torch.randn(X_train_tensor.shape[1], dtype=torch.float, requires_grad = True)
b = torch.tensor(-3.0, requires_grad=True)

y_train_tensor = torch.tensor(y_train).float()
y_test_tensor = torch.tensor(y_test).float()

learning_rate = 0.001

prediction = w @ X_train_tensor[0] + b
print(prediction)

actual = y_train_tensor[0]

loss = torch.log(1 + torch.exp(-actual * prediction))

print(loss)

loss.backward()
print(w.grad)
print(b.grad)

for epoch in range(10):  # consider each data point once in each epoch
    for i in range(len(X_train_tensor)):  # iterate over all data points
        prediction = w @ X_train_tensor[i] + b
        actual = y_train_tensor[i]
        loss = torch.log(1 + torch.exp(- actual * prediction))
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        w.grad.zero_()
        b.grad.zero_()

print(w, b)

prediction = w @ X_train_tensor[3] + b
print(prediction)

actual = y_train_tensor[3]
print(actual)