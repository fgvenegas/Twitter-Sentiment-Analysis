
# coding: utf-8

# In[1]:

import pandas as pd  
pd.options.mode.chained_assignment = None
import numpy as np  
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# On a more general level, word2vec embeds non trivial semantic and syntaxic relationships between words. This results in preserving a rich context

# In[144]:

def tokenize(tweet):
    try:
        tweet  = str(tweet).lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'
    
def tokenizing_and_cleaning(data):
    tokens = []
    for tweet in tqdm(data[:, 1]):
        tokens.append(list(tokenize(tweet)))
    tokens = np.array(tokens)
    tokens[tokens != 'NC']
    return tokens

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


# In[129]:

data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin')


# In[118]:

data_v = np.array(data)
sentiment = data_v[:, 0]
text = data_v[:, 5]
sentiment[sentiment == 4] = 1
tweets = np.c_[ sentiment, text]


# In[119]:

#n = 10000
#tweets = np.concatenate((tweets[:n], tweets[-n:]))
tweets = tweets[:1000000]
tokens = tokenizing_and_cleaning(tweets)


# In[120]:

tweets = np.c_[ tweets, tokens]


# In[121]:

x_train, x_test, y_train, y_test = train_test_split(tweets[:, 2],
                                                    tweets[:, 0], test_size=0.2)


# In[122]:

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')


# In[123]:

tweet_w2v = Word2Vec(size=200, min_count=10)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)


# In[138]:

tweet_w2v.most_similar('game')


# In[139]:

tweet_w2v.save("English_version_w2v")


# In[141]:

print ('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print ('vocab size :', len(tfidf))


# In[ ]:

from sklearn.preprocessing import scale

print("Train")
train_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in (map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)
print("Test")
test_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in (map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)


# In[ ]:

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)


# In[ ]:


score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print (score[1])

