import torch

X = torch.tensor([
    [3.0, 5.0],
    [4.0, 2.0],
    [2.0, 9.0],
    [7.0, 6.0],
    [1.0, 4.0],
    [9.0, 1.0],
])

Y = torch.tensor([22.2, 35.3, 7.5, 59.0, 5, 85])

# Initialize w1 = 2, w2 = 5, b = -3

w = torch.tensor([2.0, 5.0], requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)

# X[0] is the input [x1, x2] for the first data point
print(X[0])

# Take the dot product of w and X[0] (using the @ operator), and add b
# [w1, w2] @ [x1, x2] + b = w1*x1 + w2*x2 + b = f(x1, x2)
# That's the PREDICTION that our machine makes on the first data point

# Checkpoint: What should this print?
prediction = w @ X[0] + b
print(prediction)

actual = Y[0]

loss = (prediction - actual)**2

# Checkpoint: What should this print?
print(loss)

# This magical PyTorch function calculates the gradient!
loss.backward()
print(w.grad)
print(b.grad)

# Checkpoint: what do these gradients/derivatives mean?

# Move opposite the gradient to decrease loss
# Don't want to take huge steps, so scale down the steps with a learning rate

learning_rate = 0.001

with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

# Checkpoint: What should w1, w2, and b be now? Higher/lower than before?
print(w, b)

# We have to clear the gradients so that they can be computed from scratch in the future
w.grad.zero_()
b.grad.zero_()

# Okay, now let's repeat this many, many times, for all of the data points! :)

for epoch in range(10000):  # consider each data point once in each epoch
    for i in range(len(X)):  # iterate over all data points
        prediction = w @ X[i] + b
        actual = Y[i]
        loss = (prediction - actual)**2
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        w.grad.zero_()
        b.grad.zero_()

print(w, b)
