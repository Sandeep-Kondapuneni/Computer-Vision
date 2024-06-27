#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pytest
_ = pytest.importorskip("numpy")


# In[3]:


from nltk.metrics import *


# In[4]:


reference = 'DET NN VB DET JJ NN NN IN DET NN'.split()
test    = 'DET VB VB DET NN NN NN IN DET NN'.split()
print(accuracy(reference, test))
0.8


# In[5]:


reference_set = set(reference)
test_set = set(test)
precision(reference_set, test_set)


# In[6]:


print(recall(reference_set, test_set))


# In[7]:


print(f_measure(reference_set, test_set))


# In[9]:


from nltk import FreqDist, MLEProbDist
pdist1 = MLEProbDist(FreqDist("aldjfalskfjaldsf"))
pdist2 = MLEProbDist(FreqDist("aldjfalssjjlldss"))
print(log_likelihood(['a', 'd'], [pdist1, pdist2]))


# In[10]:


edit_distance("rain", "shine")


# In[11]:


edit_distance_align("shine", "shine")


# In[12]:


edit_distance_align("rain", "brainy")


# In[13]:


edit_distance_align("", "brainy")


# In[14]:


edit_distance_align("", "")


# In[15]:


s1 = set([1,2,3,4])
s2 = set([3,4,5])
binary_distance(s1, s2)


# In[16]:


print(jaccard_distance(s1, s2))


# In[17]:


print(masi_distance(s1, s2))


# In[18]:


spearman_correlation({'e':1, 't':2, 'a':3}, {'e':1, 'a':2, 't':3})


# In[19]:


s1 = "000100000010"
s2 = "000010000100"
s3 = "100000010000"
s4 = "000000000000"
s5 = "111111111111"
windowdiff(s1, s1, 3)


# In[20]:


abs(windowdiff(s1, s2, 3) - 0.3)  < 1e-6  # windowdiff(s1, s2, 3) == 0.3


# In[21]:


abs(windowdiff(s2, s3, 3) - 0.8)  < 1e-6  # windowdiff(s2, s3, 3) == 0.8


# In[22]:


windowdiff(s1, s4, 3)


# In[23]:


windowdiff(s1, s5, 3)


# In[24]:


reference = 'This is the reference data.  Testing 123.  aoaeoeoe'
test =      'Thos iz_the rifirenci data.  Testeng 123.  aoaeoeoe'
print(ConfusionMatrix(reference, test))


# In[25]:


cm = ConfusionMatrix(reference, test)
print(cm.pretty_format(sort_by_count=True))


# In[26]:


print(cm.pretty_format(sort_by_count=True, truncate=10))


# In[27]:


print(cm.pretty_format(sort_by_count=True, truncate=10, values_in_chart=False))


# In[28]:


cm.recall("e")


# In[29]:


cm.precision("e") 


# In[30]:


cm.f_measure("e") 


# In[31]:


n_new_companies, n_new, n_companies, N = 8, 15828, 4675, 14307668
bam = BigramAssocMeasures
bam.raw_freq(20, (42, 20), N) == 20. / N


# In[32]:


bam.student_t(n_new_companies, (n_new, n_companies), N)


# In[33]:


bam.chi_sq(n_new_companies, (n_new, n_companies), N)


# In[34]:


bam.likelihood_ratio(150, (12593, 932), N)


# In[ ]:


#For other associations, we ensure the ordering of the measures:


# In[35]:


bam.mi_like(20, (42, 20), N) > bam.mi_like(20, (41, 27), N)


# In[36]:


bam.pmi(20, (42, 20), N) > bam.pmi(20, (41, 27), N)


# In[37]:


bam.phi_sq(20, (42, 20), N) > bam.phi_sq(20, (41, 27), N)


# In[38]:


bam.poisson_stirling(20, (42, 20), N) > bam.poisson_stirling(20, (41, 27), N)


# In[39]:


bam.jaccard(20, (42, 20), N) > bam.jaccard(20, (41, 27), N)


# In[40]:


bam.dice(20, (42, 20), N) > bam.dice(20, (41, 27), N)


# In[41]:


bam.fisher(20, (42, 20), N) > bam.fisher(20, (41, 27), N) 


# In[42]:


#For trigrams, we have to provide more count information:


# In[43]:


n_w1_w2_w3 = 20
n_w1_w2, n_w1_w3, n_w2_w3 = 35, 60, 40
pair_counts = (n_w1_w2, n_w1_w3, n_w2_w3)
n_w1, n_w2, n_w3 = 100, 200, 300
uni_counts = (n_w1, n_w2, n_w3)
N = 14307668
tam = TrigramAssocMeasures
tam.raw_freq(n_w1_w2_w3, pair_counts, uni_counts, N) == 1. * n_w1_w2_w3 / N


# In[44]:


uni_counts2 = (n_w1, n_w2, 100)
tam.student_t(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.student_t(n_w1_w2_w3, pair_counts, uni_counts, N)


# In[45]:


tam.chi_sq(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.chi_sq(n_w1_w2_w3, pair_counts, uni_counts, N)


# In[46]:


tam.pmi(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.pmi(n_w1_w2_w3, pair_counts, uni_counts, N)


# In[47]:


tam.likelihood_ratio(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.likelihood_ratio(n_w1_w2_w3, pair_counts, uni_counts, N)


# In[48]:


tam.poisson_stirling(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.poisson_stirling(n_w1_w2_w3, pair_counts, uni_counts, N)


# In[49]:


tam.jaccard(n_w1_w2_w3, pair_counts, uni_counts2, N) > tam.jaccard(n_w1_w2_w3, pair_counts, uni_counts, N)


# In[50]:


#For fourgrams, we have to provide more count information:


# In[51]:


n_w1_w2_w3_w4 = 5
n_w1_w2, n_w1_w3, n_w2_w3 = 35, 60, 40
n_w1_w2_w3, n_w2_w3_w4 = 20, 10
pair_counts = (n_w1_w2, n_w1_w3, n_w2_w3)
triplet_counts = (n_w1_w2_w3, n_w2_w3_w4)
n_w1, n_w2, n_w3, n_w4 = 100, 200, 300, 400
uni_counts = (n_w1, n_w2, n_w3, n_w4)
N = 14307668
qam = QuadgramAssocMeasures
qam.raw_freq(n_w1_w2_w3_w4, pair_counts, triplet_counts, uni_counts, N) == 1. * n_w1_w2_w3_w4 / N


# In[56]:


import nltk
from nltk import word_tokenize



# In[60]:


s1= (" i am sandeep kishan from telangana")


# In[61]:


s1


# In[62]:


word_tokenize(s1)


# In[ ]:




