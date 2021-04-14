#!/usr/bin/env python
# coding: utf-8

# In[1]:


import runClassifier


# In[2]:


import linear


# In[3]:


import datasets


# In[4]:


import mlGraphics


# In[5]:


f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 0, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)


# In[6]:


f


# In[7]:


mlGraphics.plotLinearClassifier(f, datasets.TwoDAxisAligned.X, datasets.TwoDAxisAligned.Y)


# In[8]:


f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})


# In[9]:


runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)


# In[10]:


f


# In[11]:


f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})


# In[12]:


runClassifier.trainTestSet(f, datasets.TwoDDiagonal)


# In[13]:


f


# In[14]:


f = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})


# In[15]:


runClassifier.trainTestSet(f, datasets.TwoDDiagonal)


# In[16]:


f


# In[17]:


sq = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(sq, datasets.WineDataBinary)


# In[18]:


hinge = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(hinge, datasets.WineDataBinary)


# In[19]:


logLoss = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(logLoss, datasets.WineDataBinary)


# In[20]:


# (datasets.WineDataBinary().words)


# In[21]:


# logLoss


# In[22]:


keys = (datasets.WineDataBinary().words)
values = logLoss.getRepresentation()
wt_words = dict(zip(keys, values))
# wt_words


# In[33]:


# for w in sorted(wt_words, key=wt_words.get, reverse=True):
    # print(w, wt_words[w])


# In[41]:

print("Top 5 words with greatest positive weights:\n")
i = 0
for w in sorted(wt_words, key=wt_words.get, reverse=True):
    if i <5:
        print("Word -> ", w, " Weight -> ", wt_words[w])
        i = i+1
    


# In[42]:


print ("Top 5 words with greatest negative weights (in reverse order):\n")
i = 0
for w in sorted(wt_words, key=wt_words.get, reverse=True):
    if (i >= (len(keys)-5)) and (i < len(keys)):
        print("Word -> ", w, " Weight -> ", wt_words[w])
        i = i+1
    else:
        i = i+1


# In[ ]:




