"""
Implementation of *regularized* linear classification/regression by
plug-and-play loss functions
"""

from numpy import *
from pylab import *

from binary import *
from gd import *

class LossFunction:
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """

        util.raiseNotDefined()

    def lossGradient(self, X, Y, Yhat):
        """
        The inputs are in the matrix X, the true values are in the
        vector Y; the predicted values are in Yhat; compute the
        gradient of the loss associated with these predictions.
        """

        util.raiseNotDefined()


class SquaredLoss(LossFunction):
    """
    Squared loss is (1/2) * sum_n (y_n - y'_n)^2
    """

    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """

        return 0.5 * dot(Y - Yhat, Y - Yhat)


    def lossGradient(self, X, Y, Yhat):
        """
        The inputs are in the matrix X, the true values are in the
        vector Y; the predicted values are in Yhat; compute the
        gradient of the loss associated with these predictions.
        """

        return - sum((Y - Yhat) * X.T, axis=1)


class LogisticLoss(LossFunction):
    """
    Logistic loss is sum_n log(1 + exp(- y_n * y'_n))
    """

    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        
        # log[1+exp(-y_n*(w^T*x_n))]
        return sum(log(1+exp(-Y*Yhat)))


    def lossGradient(self, X, Y, Yhat):
        """
        The inputs are in the matrix X, the true values are in the
        vector Y; the predicted values are in Yhat; compute the
        gradient of the loss associated with these predictions.
        """
        SubGradient_logloss = 0
        # Chain rule
        num = exp(-Y*Yhat)*-Y*X.T
        denom = (1+exp(-Y*Yhat))
        SubGradient_logloss = sum(num/denom, axis=1)
        return SubGradient_logloss
        

class HingeLoss(LossFunction):
    """
    Hinge loss is sum_n max{ 0, 1 - y_n * y'_n }
    """

    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """

        #max{0, 1-y_n*(w^T*x_n)}
        h_loss = 1-(Y*Yhat)
        sum_loss = 0
        for i in range(0, len(Y)):
            if h_loss[i] > 0:
                sum_loss = sum_loss + h_loss[i]
        return sum_loss

    def lossGradient(self, X, Y, Yhat):
        """
        The inputs are in the matrix X, the true values are in the
        vector Y; the predicted values are in Yhat; compute the
        gradient of the loss associated with these predictions.
        """
        # SubGradient = 0 if y_n*(w.x_n+b) > 1 and...
        # SubGradient = -y_n*x_n otherwise
        SubGradient_hloss = 0
        for i in range(0, len(Y)):
            if Y[i]*Yhat[i]>1:
                SubGradient_hloss += 0
            else:
                SubGradient_hloss += -Y[i]*X[i]
        return SubGradient_hloss
         

class LinearClassifier(BinaryClassifier):
    """
    This class defines an arbitrary linear classifier parameterized by
    a loss function and a ||w||^2 regularizer.
    """

    def __init__(self, opts):
        """
        Initialize the classifier.  Like perceptron, we need to start
        out with a weight vector; unlike perceptron, we'll leave off
        the bias.  Also, we are not online, so we implement that full
        train method.
        """

        # remember the options
        self.opts = opts

        # just call reset
        self.reset()

    def reset(self):
        self.weights = 0

    def online(self):
        """
        We're not online
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return    "w=" + repr(self.weights)

    def predict(self, X):
        """
        X is a vector that we're supposed to make a prediction about.
        Our return value should be the margin at this point.
        Semantically, a return value <0 means class -1 and a return
        value >=0 means class +1
        """

        if type(self.weights) == int:
            return 0
        else:
            return dot(self.weights, X)

    def getRepresentation(self):
        """
        Return the weights
        """
        return self.weights

    def train(self, X, Y):
        """
        Train a linear model using gradient descent, based on code in
        module gd.
        """

        # get the relevant options
        lossFn   = self.opts['lossFunction']         # loss function to optimize
        lambd    = self.opts['lambda']               # regularizer is (lambd / 2) * ||w||^2
        numIter  = self.opts['numIter']              # how many iterations of gd to run
        stepSize = self.opts['stepSize']             # what should be our GD step size?

        # define our objective function based on loss, lambd and (X,Y)
        def func(w):
            # should compute obj = loss(w) + (lambd/2) * norm(w)^2
            # Yhat = Y' = wT*x_n+b             
            Yhat = sum((w*X), axis=1)
            obj  = lossFn.loss(Y,Yhat) + ((lambd/2.0) * (norm(w)**2))

            # return the objective
            return obj

        # define our gradient function based on loss, lambd and (X,Y)
        def grad(w):
            # should compute gr = grad(w) + lambd * w
            # Yhat = Y' = wT*x_n+b
            Yhat = sum((w*X), axis=1)

            gr   = lossFn.lossGradient(X,Y,Yhat) + lambd * w

            return gr

        # run gradient descent; our initial point will just be our
        # weight vector
        w, trajectory = gd(func, grad, self.weights, numIter, stepSize)

        # store the weights and trajectory
        self.weights = w
        self.trajectory = trajectory