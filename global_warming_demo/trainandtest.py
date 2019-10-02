import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('global_warming_tweets.csv',encoding = "utf-8-sig",engine='python',error_bad_lines=False)

# play around with the dataset to see what data it contains
print(df)
print('Tweet:', df['tweet'][16], '\nExistence:', df['existence'][16])
print()
print('Tweet:', df['tweet'][17], '\nExistence:', df['existence'][17])

# feel free to comment out the print statements above once you're done examining the df

# Checkpoint: what does existence=1 mean? what does existence=0 mean?

# Checkpoint: why do we split our dataset into a training set and a testing set?
train_df, test_df = train_test_split(df, test_size=0.1)

vectorizer = CountVectorizer()      # Bag-of-words vectorizer; counts occurences of each word in each tweet
vectorizer.fit(train_df['tweet'])   # Fitting the vectorizer = create a vocabulary to index mapping

"""
You can also try using different types of vectorizers (tfidf, bigram). See links below:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
https://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage

Or you could even try using your own bag-of-words implementation from a few weeks ago!
"""

def tweets_to_tensor(tweet_list):
    """
    Takes a list of n tweets and converts each tweet into a vector with d dimensions.
    Return a (n x d) torch tensor (each row is a d-dimensional vector representing one tweet)
    """
    
    X = vectorizer.transform(tweet_list)

    X = X.todense()
    # the toarray method converts X from a sparse array to a normal (dense) numpy array
    # a sparse array contains mostly 0s, so it can be stored as a Map from
    # (row, col) coordinates to the value at that coordinate, skipping over all values that are 0.
    # This can save a lot of space (since we don't have to store 0s), but it's not compatible with pytorch

    X = torch.tensor(X, dtype=torch.float)
    return X

X_train = tweets_to_tensor(train_df['tweet'])
Y_train = train_df['existence'].values

# Checkpoint: what do the dimensions of X_train represent?
print("Shape of X_train:", X_train.shape)

from model import TweetClassifier
# Go to model.py and implement the TweetClassifier class!

# Checkpoint: the constructor for TweetClassifier needs to know how many dimensions the weight
# vector should have. Pass that in as an argument.
classifier = TweetClassifier(NUM_DIMS_OF_WEIGHT_VECTOR)

# the optimizer can take care of gradient descent for us (we set its learning rate to 0.01)
# SGD stands for stochastic gradient descent
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)

for epoch in range(50):
    print("Epoch:", epoch)

    for i in range(len(X_train)):
        prediction = classifier.forward(X_train[i])

        # Checkpoint: Write a boolean expression determining if the tweet believes in global warming
        # (based on the actual label, not the prediction)
        if TWEET_BELIEVES_IN_GLOBAL_WARMING:
            # Checkpoint: Write the loss function for a tweet that believes in global warming 
            loss = INSERT_LOSS_FUNCTION
        else:
            # Checkpoint: Write the loss function for a tweet that does not believe in global warming 
            loss = INSERT_LOSS_FUNCTION

        # Sanity Check:
        # The loss should approach 0 as the predictor becomes more confident in the correct label
        # approach infinity as the predictor becomes more confident in the wrong label

        loss.backward()         # calculate the gradients
        optimizer.step()        # take step down the gradient
        optimizer.zero_grad()   # reset the gradients to zero


# now the model is done training, time to test it!
correct = 0
file = open('testpredictions.txt', 'w')
X_test = tweets_to_tensor(test_df['tweet'])
Y_test = test_df['existence'].values

for i in range(len(X_test)):
    prediction = classifier.forward(X_test[i])

    print(test_df['tweet'].values[i], file=file)
    print("Prediction:", float(prediction), "\tActual:", Y_test[i], file=file)
    print("----------------------------", file=file)

    # Checkpoint: Write a boolean expression that determines if the prediction matches the actual label
    if PREDICTION_MATCHES_ACTUAL_LABEL:
        correct += 1

print("Accuracy", correct / len(X_test))

# Go look at testpredictions.txt to see individual predictions!
# Might provide insight on why the predictor messes up on specific examples

# Examine weights on specific words
words_to_examine = ["conspiracy", "environment"]

for word in words_to_examine:
    word_index = vectorizer.get_feature_names().index(word)
    print("Word:", word, "\tCorresponding weight:", classifier.weight_vec[word_index])

# Checkpoint: how can we interpret the weights corresponding to "conspiracy" and "environment"?