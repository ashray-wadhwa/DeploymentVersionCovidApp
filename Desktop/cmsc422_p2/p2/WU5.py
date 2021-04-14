#!/usr/bin/env python
# coding: utf-8




import runClassifier





import linear





import datasets





import mlGraphics





f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 0, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)





f





mlGraphics.plotLinearClassifier(f, datasets.TwoDAxisAligned.X, datasets.TwoDAxisAligned.Y)





f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})





runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)





f





f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})





runClassifier.trainTestSet(f, datasets.TwoDDiagonal)





f





f = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})





runClassifier.trainTestSet(f, datasets.TwoDDiagonal)





f





sq = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(sq, datasets.WineDataBinary)





hinge = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(hinge, datasets.WineDataBinary)





logLoss = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(logLoss, datasets.WineDataBinary)





# (datasets.WineDataBinary().words)




# logLoss





keys = (datasets.WineDataBinary().words)
values = logLoss.getRepresentation()
wt_words = dict(zip(keys, values))
# wt_words





# for w in sorted(wt_words, key=wt_words.get, reverse=True):
    # print(w, wt_words[w])




print("\nTop 5 words with greatest positive weights:\n")
i = 0
for w in sorted(wt_words, key=wt_words.get, reverse=True):
    if i <5:
        print("Word -> ", w, " Weight -> ", wt_words[w])
        i = i+1
    





print ("\nTop 5 words with greatest negative weights (in reverse order):\n")
i = 0
for w in sorted(wt_words, key=wt_words.get, reverse=True):
    if (i >= (len(keys)-5)) and (i < len(keys)):
        print("Word -> ", w, " Weight -> ", wt_words[w])
        i = i+1
    else:
        i = i+1







