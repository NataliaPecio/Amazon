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
from sklearn.metrics import roc_curve, roc_auc_score,
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#wczytanie
dane = pd.read_csv("allreviews.csv")
reviews = dane
reviews = reviews.drop(["ReviewTitle"], axis=1)
reviews = reviews.drop(["Product"], axis=1)
reviews = reviews.drop_duplicates(keep='first')
#Sentiment Analysis
reviews["Polarity"] = reviews["ReviewBody"].apply(lambda x: TextBlob(x).sentiment[0])
reviews["PolarityB"] = np.where(reviews["Polarity"]>0.2,1,(np.where(reviews["Polarity"]<-0.2,0,100)))
# przedział neutralnych -0.2 do 0.2, bo to 1/5, bo 5 gwiazdek
pd.crosstab(index = reviews["PolarityB"], columns="Total count")
#Positive 58%
#Negative 7%
#Neutral  35%
#Positive&Negative
reviews["Satisfied"] = np.where(reviews["ReviewStar"]>3,1,(np.where(reviews["ReviewStar"]<3,0,100)))
pd.crosstab(index = reviews["Satisfied"], columns="Total count")
#Positive 64%
#Negative 25%
#Neutral 11%
reviews["Difference"]= reviews["Satisfied"]-reviews["PolarityB"]
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
#usuwam neutralne
reviews = reviews[reviews["ReviewStar"]!=3]
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
otherwords = ["amazon","jbl", "sennheiser", "boat", "bought","buy","purchase","purchasing", "product","earphone","earphones","ear","headphone","headphones","music","bluetooth"]
reviews["ReviewBody"] = reviews["ReviewBody"].apply(lambda x: ' '.join([word for word in x.split() if word not in (otherwords)]))
#stemming 
ps = PorterStemmer()
reviews["ReviewBody"] = reviews["ReviewBody"].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
#class proportion - 72% i 28% 
pd.crosstab(index = reviews["Satisfied"], columns="Total count")
example = ["phones", "satisfied", "incredibly"]
for word in example:
    print(ps.stem(word))
######## WORDLIST ##########
wordlist = pd.Series(np.concatenate([x.split() for x in reviews.ReviewBody])).value_counts() 
wordlist25 = wordlist.head(25)
wordlist25.plot.barh(width=0.5,fontsize=20).invert_yaxis()
plt.savefig("C:/Users/Natalia/Documents/Python/Amazon/plt1.jpg")
########## WORDCLOUD #######
fulltext = " ".join(r for r in reviews.ReviewBody)
wordcloud = WordCloud(background_color="white").generate(fulltext)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("C:/Users/Natalia/Documents/Python/Amazon/wc1.jpg")
############ WORDCLOUD mask #######
chmura_mask = np.array(Image.open("C:/Users/Natalia/Documents/Python/Amazon/chmura.png"))
wordcloud = WordCloud(background_color="white", max_words=250, mask=chmura_mask,contour_width=2,contour_color="navy", collocations=True).generate(fulltext)
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
wordlistpos25 = wordlistpos.head(25)
wordlistpos25.plot.barh(color="green", width=0.5,fontsize=20).invert_yaxis()
plt.savefig("C:/Users/Natalia/Documents/Python/Amazon/pltpos.jpg")
#POSITIVE WORDCLOUD
fulltextpos = " ".join(r for r in revpositive.ReviewBody)
happy_mask = np.array(Image.open("C:/Users/Natalia/Documents/Python/Amazon/jeden.png"))
wordcloud = WordCloud(background_color="white", max_words=250, mask=happy_mask,contour_width=2,contour_color="green", collocations=True).generate(fulltextpos)
wordcloud.to_file("C:/Users/Natalia/Documents/Python/Amazon/tohappy.png")
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#NEGATIVE WORDLIST
wordlistneg = pd.Series(np.concatenate([x.split() for x in revnegative.ReviewBody])).value_counts() 
wordlistneg25 = wordlistneg.head(25)
wordlistneg25.plot.barh(color="red",width=0.5, fontsize=20).invert_yaxis()
plt.savefig("C:/Users/Natalia/Documents/Python/Amazon/pltneg.jpg")
#NEGATIVE WORDCLOUD
fulltextneg = " ".join(r for r in revnegative.ReviewBody)
sad_mask = np.array(Image.open("C:/Users/Natalia/Documents/Python/Amazon/dwa.png"))
wordcloud = WordCloud(background_color="white", max_words=250, mask=sad_mask,contour_width=2,contour_color="red", collocations=True).generate(fulltextneg)
wordcloud.to_file("C:/Users/Natalia/Documents/Python/Amazon/tosad.png")
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#WYJĄTKOWO NIEZADOWOLENI KLIENCI, RATE = 1
revextnegative = reviews[reviews.ReviewStar == 1]
wordlistextneg = pd.Series(np.concatenate([x.split() for x in revextnegative.ReviewBody])).value_counts() 
wordlistextneg25 = wordlistextneg.head(25)
wordlistextneg25.plot.barh(color="black",width=0.5,fontsize=20).invert_yaxis()
plt.show()
# Subjectivity
reviews["Subjectivity"] = reviews["ReviewBody"].apply(lambda x: TextBlob(x).sentiment[1])
reviews["SubjectivityB"] = np.where(reviews["Subjectivity"]>0.666,1,(np.where(reviews["Subjectivity"]<0.333,0,100)))
# przedział 0.333-0.666 neutralne - trudno powiedzieć
pd.crosstab(index = reviews["SubjectivityB"], columns="Total count")
pd.crosstab(index = reviews["SubjectivityB"], columns="Total count").plot(kind='pie', subplots=True, labels=('Objektywne','Subjektywne','Trudno powiedzieć'),autopct='%1.1f%%')
#subjektywne 1493; 71% poz, 29% neg
#obiektywne 2873 58% poz, 42% neg
#pozytywne 8362 - SUB 13%, OB 20%
#negatywne 3281 - SUB 13%, OB 37%
# Pozytywne i Subjektywne 9%
reviews[(reviews.Satisfied==1) & (reviews.SubjectivityB==1)].count()
# Negatywne i Subjektywne 4%
reviews[(reviews.Satisfied==0) & (reviews.SubjectivityB==1)].count()
# Pozytywne i Obiektywne  14%
reviews[(reviews.Satisfied==1) & (reviews.SubjectivityB==0)].count()
# Negatywne i Obiektywne  10%
reviews[(reviews.Satisfied==0) & (reviews.SubjectivityB==0)].count()
# podział na train i test - random split
X_train, X_test, y_train, y_test = train_test_split(reviews['ReviewBody'], reviews['Satisfied'], random_state=0)
#Model LG with CountVectorizer
vect = CountVectorizer().fit(X_train)
vect.get_feature_names()[::500] 
print(vect.vocabulary_)
X_train_vectorized = vect.transform(X_train)
print(X_train_vectorized[8725])
print('Shape', X_train_vectorized.shape) 
print(type(X_train_vectorized))
print(X_train_vectorized.toarray()[600])
#logistic regression
model = LogisticRegression(multi_class='ovr',n_jobs=1,solver='liblinear')
model.fit(X_train_vectorized,y_train)
# testing
predictions = model.predict(vect.transform(X_test))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
predictions_zero = [0 for _ in range(len(y_test))]
auc_lr = roc_auc_score(y_test, predictions)
auc_zero =roc_auc_score(y_test,predictions_zero)
print('AUC: ',roc_auc_score(y_test, predictions))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions)
false_positive_rate_zero, true_positive_rate_zero, thresholds = roc_curve(y_test, predictions_zero)
#roc_auc = auc(false_positive_rate, true_positive_rate)
#plotting
plt.title("Model")
plt.plot(false_positive_rate,true_positive_rate,'r',label="AUC = %0.3f"% auc_lr)
plt.legend(loc='lower right')
plt.plot(false_positive_rate_zero,true_positive_rate_zero,'b--',label="AUC = %0.2f"% auc_zero)
plt.ylabel('"True Positive"', fontsize=12)
plt.xlabel('"False Positive"', fontsize=12)
plt.savefig("C:/Users/Natalia/Documents/Python/Amazon/plot1.jpg")
################################################ Model LG with Tf-Idf Vectorizer
vectTf = TfidfVectorizer(smooth_idf=False).fit(X_train)
print(vectTf.vocabulary_) 
print(vectTf.idf_)
X_train_vectorizedTfidf = vectTf.transform(X_train)
print(X_train_vectorizedTfidf)
model = LogisticRegression(multi_class='ovr',n_jobs=1,solver="liblinear")
model.fit(X_train_vectorizedTfidf,y_train)
predictionsTfidf = model.predict(vectTf.transform(X_test))
print(confusion_matrix(y_test,predictionsTfidf))
print(classification_report(y_test,predictionsTfidf))
print("AUC: ", roc_auc_score(y_test,predictionsTfidf))
auc_lr_tfidf = roc_auc_score(y_test, predictionsTfidf)
false_positive_rate_tfidf, true_positive_rate_tfidf, thresholds = roc_curve(y_test, predictionsTfidf)
false_positive_rate_zero, true_positive_rate_zero, thresholds = roc_curve(y_test, predictions_zero)
plt.title("Model")
plt.plot(false_positive_rate_tfidf,true_positive_rate_tfidf,'r',label="AUC = %0.3f"% auc_lr_tfidf)
plt.legend(loc='lower right')
plt.plot(false_positive_rate_zero,true_positive_rate_zero,'b--',label="AUC = %0.3f"% auc_zero)
plt.ylabel('"True Positive"', fontsize=12)
plt.xlabel('"False Positive"', fontsize=12)
plt.savefig("C:/Users/Natalia/Documents/Python/Amazon/plotTfidf.jpg")
####################################### Model Random Forest with CountVectorizer
model = RandomForestClassifier(n_estimators=1000, random_state=0) 
model.fit(X_train_vectorized,y_train)
predictionsRF = model.predict(vect.transform(X_test))
print(confusion_matrix(y_test,predictionsRF))
print(classification_report(y_test,predictionsRF))
print("AUC: ", roc_auc_score(y_test,predictionsRF))
false_positive_rate_RF, true_positive_rate_RF, thresholds = roc_curve(y_test, predictionsRF)
#roc_auc = auc(false_positive_rate, true_positive_rate)
#plotting
auc_RF = roc_auc_score(y_test, predictionsRF)
plt.title("Model")
plt.plot(false_positive_rate_RF,true_positive_rate_RF,'r',label="AUC = %0.3f"% auc_RF)
plt.legend(loc='lower right')
plt.plot(false_positive_rate_zero,true_positive_rate_zero,'b--',label="AUC = %0.3f"% auc_zero)
plt.ylabel('"True Positive"', fontsize=12)
plt.xlabel('"False Positive"', fontsize=12)
plt.savefig("C:/Users/Natalia/Documents/Python/Amazon/plotRF.jpg")
#################################### MODEL SVM
model = SVC(kernel='linear')
model.fit(X_train_vectorized,y_train)
predictionsSVM = model.predict(vect.transform(X_test))
print(confusion_matrix(y_test,predictionsSVM))
print(classification_report(y_test,predictionsSVM))
###
auc_SVM = roc_auc_score(y_test, predictionsSVM)
false_positive_rate_SVM, true_positive_rate_SVM, thresholds = roc_curve(y_test, predictionsSVM)
plt.title("Model")
plt.plot(false_positive_rate_SVM,true_positive_rate_SVM,'r',label="AUC = %0.3f"% auc_SVM)
plt.legend(loc='lower right')
plt.plot(false_positive_rate_zero,true_positive_rate_zero,'b--',label="AUC = %0.3f"% auc_zero)
plt.ylabel('"True Positive"', fontsize=12)
plt.xlabel('"False Positive"', fontsize=12)
plt.savefig("C:/Users/Natalia/Documents/Python/Amazon/plotSVM.jpg")

