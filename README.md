# UnsupervisedSentimentAnalysis
- Uses the famous "Sentiment140 dataset with 1.6 million tweets" dataset from Kaggle.
  - It contains the following 6 fields:

    target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

    ids: The id of the tweet ( 2087)

    date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

    flag: The query (lyx). If there is no query, then this value is NO_QUERY.

    user: the user that tweeted (robotickilldozr)

    text: the text of the tweet (Lyx is cool)

- Uses Word2vec and Clustring to classify tweets to [0, 2, 4], negative [0], neutral [2], and positive [4] sentiment.

- Uses Spacy for lemmatization and with some list comprehension to pre-process the data.

# Training:
- Split the data to train and test.
- deleted all attributes from the training set, except for text ofcourse!

- With Word2Vec I also use skipgram to build a vocabulary, Wrod2vec in simple terms embeds each word in a tweet to a vector of number for all tweets in the dataset, and skipgrams is used to give context e.g. words that come together are given a close vector representations.

- Uses K-means as the clustring algorithm, to cluster/divide words based on the vector representations.

# Testing:
*** I've read a lot about how clustring is only an explorative approach, and in case I want to classify I should use a classifier, but I choose to go with the following test model, it isn't a great model, accuracy is only 53%, but it's something that I think was worth a try, and I am kinda happy with the approach. 
So here we go! : 
- Build a new Wrod2vec model with tweets from the test set, but in the train part I use "total_examples = old_model.corpus_count" , that will result in building a new vocab from the test set.
- Initialize a new dataframe "called words" for words in the vocab, in the following structure:
    words , vecotors, cluster
   - words: the vocabulary we built from the new word2vec model.
   - vectors : vector representation for each word.
   - cluster : Uses the k-means model that was trained on the training phase to predict which cluster do each word
                    belongs to, and assign the cluster number to that word.
- Now the part where it isn't very effiecnt, I loop through the test data again, then:
    For each tweet:
   - check if it's words is in the words dataframe, if yes!, I lookup it's cluster number and append it to
      a list.
   - get the most frequent cluster number in that list.
   - I then make a prediction based on that number, e.g. if most words in that tweet belong to the negative cluster,
      then it most probably a negative tweet.
   - I compare my prediction to the actual target.
   - After looping through all the tweets in the test set, I calculate the accuray of the prediction!

Reference I used to build the model:
- https://phdstatsphys.wordpress.com/2018/12/27/word2vec-how-to-train-and-update-it/
- https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
- https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
- https://pythonprogramminglanguage.com/kmeans-text-clustering/
- https://github.com/rafaljanwojcik/Unsupervised-Sentiment-Analysis
- https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
