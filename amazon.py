#Created by Chandrashekhar Kalnad on 9/11/2019
#This code perform sentiment analysis of customer feedback i.e. It classifies feedback as positive/negative/neutral. This program also gives us the top 25 most used words
#It draws a bar chart and pie chart of these words

import pandas as pd                 #For data analysis
from textblob import TextBlob       #For sentiment analysis
from itertools import islice
import matplotlib.pyplot as plt     #For plotting
import seaborn as sns
from numpy.random import randn
from numpy.random import seed
from numpy import cov
from scipy.stats import pearsonr
from pylab import rcParams
import nltk                        #For natural language processing
from collections import Counter
from nltk.corpus import stopwords
import matplotlib as mpl

#Convert file in dataframe object
df_amazon_data = pd.read_csv("amazon_co-ecommerce_sample.csv" , low_memory = False)

#Column names for new file
COLS = ['manufacturer','text', 'sentiment','subjectivity','polarity']

df = pd.DataFrame(columns=COLS)

#Code for sentiment creation
for index, row in islice(df_amazon_data.iterrows(), 0, None):

     new_entry = []
     text_lower=str.lower(str(row['customer_reviews']))
     blob = TextBlob(text_lower)
     sentiment = blob.sentiment
		 
     polarity = sentiment.polarity
     subjectivity = sentiment.subjectivity
		 
     new_entry += [row['manufacturer'],text_lower,sentiment,subjectivity,polarity]
        
     sentimet_df = pd.DataFrame([new_entry], columns=COLS)
     df = df.append(sentimet_df, ignore_index=True)
		 
df.to_csv('Sentiment_Values.csv', mode='w', columns=COLS, index=False, encoding="utf-8")

#Filter out neutral values
dffilter = df.loc[(df.loc[:, df.dtypes != object] != 0).any(1)]

#Plot boxplot for subjectivity and polarity and save as pdf
boxplot = dffilter.boxplot(column=['subjectivity','polarity'], fontsize = 15, grid = True, vert=True, figsize=(12,12,))
plt.ylabel('Range')

plot_file_name="boxplot_amazon.pdf"
 
boxplot.figure.savefig(plot_file_name,format='pdf',dpi=100)

#Plot scatterplot for subjectivity and polarity and save as pdf
scatterplot = sns.lmplot(x='subjectivity',y='polarity',data=dffilter, fit_reg=True, scatter=True, height=10, palette="mute")

plot_file_name="scatterplot_amazon.pdf"
 
scatterplot.savefig(plot_file_name,format='pdf',dpi=100)

# prepare data
data1 = dffilter['subjectivity']
data2 = data1 + dffilter['polarity']

# calculate covariance and correlation
covariance = cov(data1, data2) 
print(covariance)

corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.5f' % corr)

#Calculate polarity Distribution for dffilter, plot histogram and save as pdf

plt.clf()
sns.distplot(dffilter['polarity'], hist=True, kde=True, bins=int(30), color = 'darkred', hist_kws={'edgecolor':'black'}, axlabel ='Polarity')
plt.title('Polarity Density')

rcParams['figure.figsize'] = 10,15

plot_file_name="histogram_amazon.pdf"
 
plt.savefig(plot_file_name,format='pdf',dpi=100)

stopwords = nltk.corpus.stopwords.words('english')

RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
words = (df.text.str.lower().replace([r'\|',r'\&',r'\-',r'\.',r'\,',r'\'',r'//',r'50',r'2015',r'nov',r'2014',r'2013',r'jan',r'dec',r'stars',r'five',r'2016',r'one',r'oct',r'feb',r'mar',r'aug',r'sept',r'july',r'may',r'april',r'2012',r'jun',r'year',r'2',r'3',r'1',r'0',r'4',r'5'r'6',r'7',r'9',r'8',r'5',r'6',r'!',RE_stopwords],
                                     ['','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','',''], regex=True).str.cat(sep=' ').split())

# generate DF out of Counter
rslt = pd.DataFrame(Counter(words).most_common(25), columns=['Word', 'Frequency']).set_index('Word')
print(rslt)

rslt_wordcloud = pd.DataFrame(Counter(words).most_common(100),
                    columns=['Word', 'Frequency'])

plt.clf()

#Draw bar chart and save as pdf
rslt.plot.bar(rot=40, figsize=(16,10), width=0.8,colormap='tab10')
plt.title("Commanly used words")
plt.ylabel("Count")

from pylab import rcParams
rcParams['figure.figsize'] = 10,15

plot_file_name="Barchart_amazon.pdf"
 
plt.savefig(plot_file_name,format='pdf',dpi=100)

plt.clf()

#Draw pie chart and save as pdf

explode = (0.2, 0.2, 0.1, 0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  
labels=['great','good','well','quality','bought','would','little','really','fun','old','product','loves','love','game','toy','set','excellent','get','nice','like','son','made','time','price','buy']

plt.pie(rslt['Frequency'], explode=explode,labels =labels , autopct='%1.1f%%', shadow=False, startangle=90)
plt.legend( labels, loc='lower left',fontsize='x-small',markerfirst = True)
plt.tight_layout()
plt.title("Commanly used words by Clients")

mpl.rcParams['font.size'] = 15.0

plot_file_name="Piechart_amazon.pdf"
 
plt.savefig(plot_file_name,format='pdf',dpi=100)
