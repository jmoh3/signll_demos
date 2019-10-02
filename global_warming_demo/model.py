import torch

class TweetClassifier(torch.nn.Module):
    """
    Object oriented programming in Python! TweetClassifier inherits from torch.nn.Module
    Encapsulating the prediction function within an object is a super useful abstraction
    especially as our predictors become more and more complex (e.g. neural nets)
    """

    def __init__(self, num_dims):
        """
        This is a constructor!

        The argument self indicates that this is an instance method, not a static method.
        num_dims is the number of dimensions in the weight vector
        """
        super(TweetClassifier, self).__init__() # this calls the constructor of the parent class

        # torch.nn.Parameter tells the module to keep track of this vector's gradient so
        # it can be updated during gradient descent

        self.weight_vec = torch.nn.Parameter(torch.randn(num_dims))
        self.bias = torch.nn.Parameter(torch.randn(1)) # bias is a vector with a single element

    def forward(self, x):
        """
        x is the vector representation of a single tweet
        This method should make a prediction on the tweet and return that number
        (More positive => stronger belief in global warming, and vice versa)
        """

        # Checkpoint: write a mathematical expression that makes a prediction on the tweet vector
        return self.weight_vec @ x + self.bias