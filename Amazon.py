# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:46:56 2019

@author: Natalia
"""

import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
# wczytanie danych, usunięcie niepotrzebnych kolumn
dane = pd.read_csv("allreviews.csv")
reviews = dane
reviews = reviews.drop(["ReviewTitle"], axis=1)
reviews = reviews.drop(["Product"], axis=1)
# Usuwanie duplikatów, oprócz pierwszego wystąpienia.
reviews = reviews.drop_duplicates(keep='first')
#Sentiment Analysis
reviews["Polarity"] = reviews["ReviewBody"].apply(lambda x: TextBlob(x).sentiment[0])
reviews["PolarityBinary"] = np.where(reviews["Polarity"]>0.2,1,(np.where(reviews["Polarity"]<-0.2,0,100)))
# przedział neutralnych -0.2 do 0.2, bo to 1/5, bo 5 gwiazdek
pd.crosstab(index = reviews["PolarityBinary"], columns="Total count")
#Positive 58%
#Negative 7%
#Neutral  35%
#Positive&Negative
reviews["Satisfied"] = np.where(reviews["ReviewStar"]>3,1,(np.where(reviews["ReviewStar"]<3,0,100)))
pd.crosstab(index = reviews["Satisfied"], columns="Total count")
#Positive 64%
#Negative 25%
#Neutral 11%
reviews["Difference"]= reviews["Satisfied"]-reviews["PolarityBinary"]
pd.crosstab(index = reviews["Difference"], columns="Total count").plot(kind='pie', subplots=True, autopct='%1.1f%%')
#  1  False positive + -                 <1%
# -1  False negative - +                 4%
#  0  Correct                           60%
# -99 Positive but Neutral + o           35% wszystkie neutral
# -100 Negative but Neutral - o
#  99 Neutral but Positive o +
#  100 Neutral but Negative o -
#false match
FalseNeg = reviews
FalseNeg = FalseNeg[FalseNeg["Difference"]==-1]
FalseNeg = FalseNeg[FalseNeg["Polarity"]>0.6] # górna 1/5 bo 5 gwiazdek
FalsePos = reviews
FalsePos = FalsePos[FalsePos["Difference"]==1]
FalsePos = FalsePos[FalsePos["Polarity"]<-0.6]
reviews = pd.concat([reviews,FalseNeg,FalsePos]).drop_duplicates(keep=False)
#usuwam duplikaty
reviews = reviews[reviews["ReviewStar"]!=3]
#usuwam neutralne zostaje 11643
# Text pre-processing
#change to lowercase
reviews["ReviewBody"] = reviews["ReviewBody"].str.lower()
#remove punctuation
def remove_punctuation(a):
    a = ''.join([i for i in a if i not in frozenset(string.punctuation)])
    return ' '+ a
reviews["ReviewBody"] = reviews["ReviewBody"].apply(remove_punctuation)
#remove emoji
def remove_emoji(a):
    emojis = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"    #signs
                           u"\U000024C2-\U0001F251"   #signs
                           "]+", flags=re.UNICODE)
    return emojis.sub(r'',a)
reviews["ReviewBody"] = reviews["ReviewBody"].apply(remove_emoji)
#remove numbers
def remove_numbers(a):
    a = ''.join([i for i in a if not i.isdigit()])
    return a
reviews["ReviewBody"] = reviews["ReviewBody"].apply(remove_numbers)
#stopwords
stopwords = stopwords.words('english')
reviews["ReviewBody"] = reviews["ReviewBody"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
####### stopwords adjustment
stopwords.extend(("also","go","went","get","getting", "got","u"))
reviews["ReviewBody"] = reviews["ReviewBody"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
#other words -> Context specific
otherwords = ["amazon","jbl", "sennheiser", "boat", "bought","buy","purchase","purchasing"]
reviews["ReviewBody"] = reviews["ReviewBody"].apply(lambda x: ' '.join([word for word in x.split() if word not in (otherwords)]))
#stemming 
ps = PorterStemmer()
reviews["ReviewBody"] = reviews["ReviewBody"].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
#class proportion - 72% i 28% OK WYNIK
pd.crosstab(index = reviews["Satisfied"], columns="Total count")
######## WORDLIST ##########
wordlist = pd.Series(np.concatenate([x.split() for x in reviews.ReviewBody])).value_counts() 
wordlist50 = wordlist.head(50)
wordlist50.plot.barh().invert_yaxis()
plt.show()
########## WORDCLOUD #######
fulltext = " ".join(r for r in reviews.ReviewBody)
wordcloud = WordCloud(background_color="white").generate(fulltext)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
############ WORDCLOUD SŁUCHAWKI #######
headphones_mask = np.array(Image.open("C:/Users/Natalia/Documents/Python/Amazon/headphones.png"))
wordcloud = WordCloud(background_color="white", max_words=250, mask=headphones_mask,contour_width=2,contour_color="white", collocations=False).generate(fulltext)
wordcloud.to_file("C:/Users/Natalia/Documents/Python/Amazon/toheadphones.png")
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
########### Z podziałem na positive, negative, omijając neutral
revpositive = reviews[reviews.Satisfied == 1]
revnegative = reviews[reviews.Satisfied == 0]
# POSITIVE WORDLIST
wordlistpos = pd.Series(np.concatenate([x.split() for x in revpositive.ReviewBody])).value_counts() 
wordlistpos50 = wordlistpos.head(50)
wordlistpos50.plot.barh(color="green").invert_yaxis()
plt.show()
#POSITIVE WORDCLOUD
fulltextpos = " ".join(r for r in revpositive.ReviewBody)
happy_mask = np.array(Image.open("C:/Users/Natalia/Documents/Python/Amazon/happy.png"))
wordcloud = WordCloud(background_color="white", max_words=250, mask=happy_mask,contour_width=2,contour_color="white", collocations=True).generate(fulltextpos)
wordcloud.to_file("C:/Users/Natalia/Documents/Python/Amazon/tohappy.png")
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#NEGATIVE WORDLIST
wordlistneg = pd.Series(np.concatenate([x.split() for x in revnegative.ReviewBody])).value_counts() 
wordlistneg50 = wordlistneg.head(50)
wordlistneg50.plot.barh(color="red", fontsize=10).invert_yaxis()
plt.show()
#NEGATIVE WORDCLOUD
fulltextneg = " ".join(r for r in revnegative.ReviewBody)
sad_mask = np.array(Image.open("C:/Users/Natalia/Documents/Python/Amazon/sad.png"))
wordcloud = WordCloud(background_color="white", max_words=250, mask=sad_mask,contour_width=2,contour_color="white", collocations=True).generate(fulltextneg)
wordcloud.to_file("C:/Users/Natalia/Documents/Python/Amazon/tosad.png")
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#WYJĄTKOWO NIEZADOWOLENI KLIENCI, RATE = 1
revextnegative = reviews[reviews.ReviewStar == 1]
wordlistextneg = pd.Series(np.concatenate([x.split() for x in revextnegative.ReviewBody])).value_counts() 
wordlistextneg50 = wordlistextneg.head(50)
wordlistextneg50.plot.barh(color="black").invert_yaxis()
plt.show()
# Subjectivity
reviews["Subjectivity"] = reviews["ReviewBody"].apply(lambda x: TextBlob(x).sentiment[1])
reviews["SubjectivityBinary"] = np.where(reviews["Subjectivity"]>0.666,1,(np.where(reviews["Subjectivity"]<0.333,0,100)))
# przedział 0.333-0.666 neutralne - trudno powiedzieć
pd.crosstab(index = reviews["SubjectivityBinary"], columns="Total count").plot(kind='pie', subplots=True, labels=('Objektywne','Subjektywne','Trudno powiedzieć'),autopct='%1.1f%%')
# Pozytywne i Subjektywne 9%
reviews[(reviews.Satisfied==1) & (reviews.SubjectivityBinary==1)].count()
# Negatywne i Subjektywne 4%
reviews[(reviews.Satisfied==0) & (reviews.SubjectivityBinary==1)].count()
# Pozytywne i Obiektywne  14%
reviews[(reviews.Satisfied==1) & (reviews.SubjectivityBinary==0)].count()
# Negatywne i Obiektywne  10%
reviews[(reviews.Satisfied==0) & (reviews.SubjectivityBinary==0)].count()
#Subjektywne - pozytwyne stanowią około 2/3, Obiektywne - pozytywne nieco ponad połowa
# podział na train i test - random split
X_train, X_test, y_train, y_test = train_test_split(reviews['ReviewBody'], reviews['Satisfied'], random_state=0)
#Model LG with CountVectorizer
vect = CountVectorizer().fit(X_train)
vect.get_feature_names()[::400] # every 400th element
print(vect.vocabulary_)
X_train_vectorized = vect.transform(X_train)
print(X_train_vectorized)
print('Shape of matrix', X_train_vectorized.shape) 
# there are A docs with B words in vocabulary
print(type(X_train_vectorized))
print(X_train_vectorized.toarray()[2])
#logistic regression
# solver - For small datasets, ‘liblinear’ is a good choice
model = LogisticRegression(multi_class='ovr',n_jobs=1,solver='liblinear')
model.fit(X_train_vectorized,y_train)
# testing the model
predictions = model.predict(vect.transform(X_test))
roc_auc_score(y_test, predictions)
print('AUC: ',roc_auc_score(y_test, predictions))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
#plotting
plt.title("Testing model")
plt.plot(false_positive_rate,true_positive_rate,'g',label="AUC = %0.2f"% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'b--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
################################################ Model LG with Tf-Idf Vectorizer
# TF = (Number of times term T appears in the particular row) / (number of terms in that row)
# 
vectTf = TfidfVectorizer(min_df=10).fit(X_train)
print(vectTf.vocabulary_) 
# in vocabulary each word has an id number assigned
# Bag-of-Words Model
# we are only concerned with encoding schemes that represent what words 
# are present (count) or the degree to which they are present (frequency) 
# in encoded documents without any information about order
print(vectTf.idf_)
# idf - the lower the score the more frequently observed the word
X_train_vectorizedTfidf = vectTf.transform(X_train)
print(X_train_vectorizedTfidf)
# nr dokumentu, nr wyrazu, wartosć scoringu
model = LogisticRegression(multi_class='ovr',n_jobs=1,solver="liblinear")
model.fit(X_train_vectorizedTfidf,y_train)
predictionsTfidf = model.predict(vectTf.transform(X_test))
print("AUC: ", roc_auc_score(y_test,predictionsTfidf))
####################################### Model Random Forest with CountVectorizer
model = RandomForestClassifier(n_estimators=1000, random_state=0) 
model.fit(X_train_vectorized,y_train)
#print(model.feature_importances_)
predictionsRF = model.predict(vect.transform(X_test))
print("AUC: ", roc_auc_score(y_test,predictionsRF))
#
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictionsRF)
roc_auc = auc(false_positive_rate, true_positive_rate)
#plotting
plt.title("Testing model")
plt.plot(false_positive_rate,true_positive_rate,'g',label="AUC = %0.2f"% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'b--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')