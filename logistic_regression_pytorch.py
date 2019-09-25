import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# load csv
df = pd.read_csv('yelp_reviews.csv')

# split up our dataset into train and test sets using a handy sklearn function
X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['pos_neg'].values, test_size=0.2)

# we're going to use sklearn's vectorizer (extra credit if you use your own)
vectorizer = CountVectorizer(lowercase=True)

# this gives us a BOW matrix for all the reviews in the training set
# we use fit transform because the words in the training set are going to make up the vocab for 
# the test set as well.
X_train_matrix = vectorizer.fit_transform(X_train)
# we use transform for x test because we don't want the vocabulary for this to be different from
# our training set
X_test_matrix = vectorizer.transform(X_test)

# Convert our matrices into float tensors for pytorch
X_train_tensor = torch.tensor(X_train_matrix.toarray(), dtype=torch.float)
X_test_tensor = torch.tensor(X_test_matrix.toarray(), dtype=torch.float)

# Initialize w and b to random values
# w's length is the length of our vocabulary (the second dimension for our train tensor)
# we must specify requires_grad=True otherwise grad will always be None (@ Jackie)
w = torch.randn(X_train_tensor.shape[1], dtype=torch.float, requires_grad = True)
b = torch.tensor(0.0, requires_grad=True)

y_train_tensor = torch.tensor(y_train, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)

learning_rate = 0.001

for epoch in range(10):  # consider each data point once in each epoch
    for i in range(len(X_train_tensor)):  # iterate over all data points
        # get our model's prediction
        prediction = w @ X_train_tensor[i] + b
        # get actual value
        actual = y_train_tensor[i]
        loss = torch.log(1 + torch.exp(- actual * prediction))
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        w.grad.zero_()
        b.grad.zero_()

# Now we calculate the accuracy of our model
correct = 0

for i in range(len(X_test_tensor)):
    prediction = w @ X_test_tensor[i] + b
    actual = y_test_tensor[i]
    # If our model predicted something greater than 0, and our actual is 1, then we increment
    # correct by 1 because our model guessed correctly.
    if (prediction > 0 and actual == 1) or (prediction < 0 and actual == -1):
        correct += 1

accuracy = correct / len(X_test_tensor)

print(f'Accuracy: {accuracy}')

# try it yourself!
to_predict = torch.tensor(vectorizer.transform(["Terrible food, terrible service. Would never reccomend to a friend"]).toarray(), dtype=torch.float)
prediction = w @ to_predict[0] + b
print(prediction)