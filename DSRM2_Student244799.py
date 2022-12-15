#!/usr/bin/env python
# coding: utf-8

# # Sussex Budget Productions: Data Insights and Reccomendations for Future Investment 2021
# 

# <b>Student Number: 244799<b>
# 
# 

# This report investigates profitability of movie genres 2000-2016 to advise Sussex Budget Productions on  recommendations for future film ventures.
# Based on the company aims to gain 1.5million funding, the potential revenue of the analysis recommendations will be projected and thus can be used in consultations with investors.<br />
# <br />
# The beginning of the report will clean and explore of movie data to propose targeted analytical ideas.
# <br />
# Based on the insights derived from the overview, hypotheses on the genres likely to bring us the most profit are constructed and tested statistically.
# <br /> 
# Following the analysis, recommendations will be stated in the summary. All analysis will be run in the Python programming language.

# ### Data Import and Cleaning

# In[12]:


#Import programming packages
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats as stats

MovieFile = 'movie_metadata.csv'

#Read movie data into DataFrame
RawData = pd.read_csv(MovieFile,index_col=None)
print(RawData.head(10))


# Below shows basic attributes of the dataset; the number of movies included, the columns (measures) available which can be used as parameters in the analysis, and the years of movie creation.

# In[3]:


#Get number of movies
Shape=RawData.shape
print("The number of movies in the dataset is: "+ str(Shape[0]))

#Print columns
print("\nThe following columns are available in the data set:\n")
for Col in RawData.columns:
    print(Col)
    
#Print Year range
YearMin=int(np.min(RawData.title_year))
YearMax=int(np.max(RawData.title_year))
print("\nThe data has been taken from the years: "+str(YearMin)+" and "+str(YearMax))


# There is plenty of information provided about the movies seen in the columns printed above, including the directors of the movie, facebook likes, countries and genres. There is also a large number of movies (5043) to analyse with release dates from 1916 to 2016. 
# 
# To keep the insights relevant to 2021 demographics and trends, only movies released after in or after 2000 will be used in the analysis.

# In[4]:


#Create subset of the data with movies from 2000 onwards
RecentData=RawData[RawData['title_year']>=2000]
RecShape=RecentData.shape
print("The number of movies in the dataset produced after 2000 are: "+str(RecShape[0]))


# ### Exploratory Data Analysis

# Next, trends of movies left in the dataset are shown by plotting the frequencies of movies 2000-2016 on bar graphs by country, content rating, genre types and year.

# In[5]:


#Plot Country Frequency
plt.subplots(2,2, figsize=(10,7))
plt.suptitle("Frequencies of Movies 2000-2016", fontsize=16)
plt.subplot(2,2,1)

xCountries=RecentData['country'].unique()
CountryFreq=RecentData.groupby('country')['imdb_score'].count()
CountryFreq=CountryFreq.sort_values(ascending=False)
CountryFreq=CountryFreq[0:10]

plt.bar(CountryFreq.index,CountryFreq)
plt.title("Frequency of Top 10 Countries")
plt.xticks(rotation=90)

#Plot Rating Frequency
plt.subplot(2,2,2)

xContentRating=RecentData['content_rating'].unique()
ContentRatingFreq=RecentData.groupby('content_rating')['imdb_score'].count()
ContentRatingFreq=ContentRatingFreq.sort_values(ascending=False)

plt.bar(ContentRatingFreq.index,ContentRatingFreq,color='lightblue')
plt.title("Frequency by Content Rating")
plt.xticks(rotation=90)

#Plot Genre Frequency
plt.subplot(2,2,3)

GenreAll = []
for line in RecentData['genres']:
    x = line.split("|")
    for m in x:
        GenreAll.append(m)
        
GenreDict=dict(sorted((x,GenreAll.count(x)) for x in set(GenreAll)))
SortedGenre=sorted(GenreDict.items(), key=lambda x: x[1],reverse=True)
SortedGenreDict={}
for item in SortedGenre:
    SortedGenreDict[item[0]]=item[1]
plt.bar(SortedGenreDict.keys(),SortedGenreDict.values(),color='cadetblue')

plt.xticks(rotation=90)
plt.title("Frequency by Genre Types")

#Plot Year Frequency
plt.subplot(2,2,4)

xYear=RecentData['title_year'].unique()
YearFreq=RecentData.groupby('title_year')['imdb_score'].count()
YearFreq=YearFreq.sort_values(ascending=False)
YearFreq.index=YearFreq.index.astype(int) 

plt.bar(sorted(YearFreq.index.astype(int)),YearFreq,color='darkslategrey')
plt.title("Frequency by Year")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# Further context is needed to show if any categories of the demographics in the above charts lead to more successful movies.
# Success will be measured by profitability. However, taking the 'gross revenue' alone misses the full picture of movie investment. Therefore, our measure of profitability will be gross revenue as a percentage of budget spent. Thus, if a film budget was 1million and gross revenue was 2million, then the gross revenue as a percentage of budget would be 200%.
# <br/><br/>
# The focus will be on the top five most frequent genres and explore if any of these show increased profitability. 
# <br/><br/>
# First, the association between budget and gross revenue will be explored to confirm if the suggested measure of gross revenue as a percentage of budget makes sense. These parameters are plotted on a scatter graph and the Pearson's coefficient, a measure which tells us the relatedness of two linear variables and is a number between 0 and 1, is calculated.
# 

# In[5]:


#Plot scatter graph for log budget vs. log gross net revenue
plt.scatter(np.log10(RecentData['budget']),np.log10(RecentData['gross']), s=1, c="darkgrey")
plt.xlabel("Log10 of Movie Budget")
plt.ylabel("Log10 of Gross Revenue of Movie")
plt.title("Relationship between Movie Budget and Gross Revenue: 2000-2016")

plt.show()


# In[6]:


#Calculate the Pearson's Coefficient
BudgRev=RecentData[['budget','gross']].dropna()
PearsonsCoeff=stats.pearsonr(np.array(np.log10(BudgRev['budget'])),np.array(np.log10(BudgRev['gross'])))
print("The Pearson's Coefficient for budget and gross revenue is: {}".format(PearsonsCoeff[1]))


# Since the coefficient is <0.05, there is a low probability that the correlation between budget and gross revenue is due to chance and therefore it is highly likely that an increased movie budget leads to an increased gross revenue. <br/><br/>
# A column is added to our dataset, giving the gross revenue as a percentage of film budget, which will be used as the measure of profitability.

# In[7]:


RecentData['Rev%Budget']=(RecentData['gross']/RecentData['budget'])*100


# ### Hypothesis Testing: Has any of the top 5 most frequent genres of movies made significantly more profit in 2000-2016?

# As seen in the exploratory data analysis, the five most frequently made genre of movie 2000-2016 made was:

# In[8]:


#Print tpp 5 movies
Top5Movies=list(SortedGenreDict.keys())[:5]
for i in Top5Movies:
    print(i)


# These will be tested to see if genre impacts profitability. If the genre of a movie impacts profitability, then movies of some genres should have a higher gross profit as a percentage of budget than movies exclusive of that genre. The hypothesis is that at least one of the five most frequently made movie genres will be significantly more profitable. The null hypothesis is that genre does not impact profitability.
# 
# Subsets of the dataset are created for each genre, and a corresponding dataset for all movies that do not contain the genre are created for comparison.

# In[9]:


Top5Data={}

for x in Top5Movies:
    (GenreData,NonGenreData)=RecentData[RecentData['genres'].str.contains(x)],RecentData[~RecentData['genres'].str.contains(x)]
    Top5Data[x]=(GenreData,NonGenreData)


# Next, a box plot and summary table is created to give an overview of the means and spread of the gross revenue as a percentage of budget.

# In[24]:


#Remove the entries where there is no value for Rev%Budget
BoxPlotData={}
for x,y in Top5Data.items():
    RemoveNaN=y[0]['Rev%Budget'][~np.isnan(y[0]['Rev%Budget'])].tolist()
    BoxPlotData[x]=RemoveNaN
BoxPlotData['All Movies']=RecentData['Rev%Budget'][~np.isnan(RecentData['Rev%Budget'])].tolist()

#Plot the boxplot
Fig,Ax=plt.subplots()
boxprops = dict(linewidth=2, color='black')
medianprops = dict(linestyle='-', linewidth=2.5, color='teal')
Ax.boxplot((np.log10(BoxPlotData['Drama']),np.log10(BoxPlotData['Comedy']),np.log10(BoxPlotData['Thriller']),np.log10(BoxPlotData['Action']),np.log10(BoxPlotData['Romance']),np.log10(BoxPlotData['All Movies'])),boxprops=boxprops,medianprops=medianprops)
Ax.set_xticklabels(BoxPlotData.keys())
Ax.set_title("Boxplots of logs of revenue as a percentage of budget, by Top 5 Genre")
Ax.set_ylabel("Log(Revenue as % of Budget)")
plt.show()

#Create summaritive table
for w,z in BoxPlotData.items():
    Series=pd.Series(z)
    BoxPlotData[w]=Series

SubDf=pd.DataFrame(BoxPlotData)
display(SubDf.describe().round(2))


# Since the maximum values seem exceptionally high, it is likely there has been some errors in data entry in the gross revenue or budget values entered. Subsequently, entries that have more than 2000% gross revenue as a percentage of budget will be removed.

# In[10]:


#Remove all entries where there is over 2000% revenue as a percentage of budget
RecentData2=RecentData[RecentData['Rev%Budget']<=2000]
Top5Data2={}
Top5DataFull={}

for x in Top5Movies:
    SubData,NotSubData=RecentData2[RecentData2['genres'].str.contains(x)],RecentData2[~RecentData2['genres'].str.contains(x)]
    SubDataS=pd.Series(SubData['Rev%Budget'])
    Top5Data2[x]=SubDataS
    Top5DataFull[x]=SubData,NotSubData

Top5Data2['All Movies']=RecentData2['Rev%Budget']
SubDf2=pd.DataFrame(Top5Data2)
display(SubDf2.describe().round(2))


# Next, each of the five genres is tested in a t-test against all other remaining movies significantly to see if there is a significant difference in profitability.

# In[11]:


#Dictionary to store P Values
GenreSigOutcomes={}

#This function gets the p value from the t test by calculating the mean, standard deviation and error
def TestSamples(G1,G2,Name):
    
    SizeGenre,SizeOther=len(G1),len(G2)
    MeanGenre,MeanOther=np.mean(G1),np.mean(G2)
    StdGenre,StdOther=np.std(G1),np.std(G2)
    ErrorGenre,ErrorOther=StdGenre/np.sqrt(G1),StdOther/np.sqrt(G2)

    GenreVsOtherTest=stats.ttest_ind_from_stats(MeanGenre,StdGenre,SizeGenre,MeanOther,StdOther,SizeOther,equal_var=False)
    OneTailDiff=GenreVsOtherTest.pvalue/2
    GenreSigOutcomes={}
    return OneTailDiff
    print ('P-value for Drama vs. Other Movies is {}'.format(OneTailDiff))

for i,j in Top5DataFull.items():
    PValue=TestSamples(j[0]['Rev%Budget'],j[1]['Rev%Budget'],i)
    GenreSigOutcomes[i]=pd.Series(PValue)
    
PvalueDf=pd.DataFrame(GenreSigOutcomes)
print("The p values for the 5 genres:")
display(PvalueDf)


# Out of the five genres tested, all but one showed a significant differences (p<0.05). This means the null hypothesis can already be rejected and that genre does impact profitability in some way. Considering the mean profitability (155.41%) of all movies, we can conclude that:
# - Dramas (p=0.233) are not significantly differential.
# - Comedies (p=0.003, mean=169.29) are a significantly more profitable genre.
# - Thrillers (p=0.003, mean=177.65) are a significantly more profitable genre.
# - Action (p=3.5 x 10^-18, mean=111.42) is a significantly less profitable genre.
# - Romance (p=0.039, mean=169.71) is a significantly more profitable genre.

# # Summary

# Taking the top five most frequent genres of movies 2000-2016, it was found that comedy, thriller and romance movies were significantly more profitable than all movies not associated with each respective genre. Therefore, investing in movie proposals within these genre domains is very much advised for increased profitability.
# If a 500,000 budget was delegated to each of these three genres, taking their mean gross revenue as a percentage of budget, an expected 2.6million gross profit is expected (500,000 x 1.6929 + 500,000 x 1.7765 + 500,000 x 1.6971), a net gain of 1.1million.
# <br/><br/>
# Contrastingly, drama movies showed no difference in profitability compared with non-dramas and action movies were significantly less profitable than non-action movies. It should be noted that the means of each of these over 100% means they are still on average profitable investments, but seem to demand more budget that does not lead to profitable outcomes.
# <br/><br/>
# There is opportunity for deeper and more accurate analysis surrounding genre based profitability. For example, reducing the number of genres listed per film since some in the dataset listed many genres. However conclusively comedy, thriller and romance have proved the most profitable and action movies the least.
# 
# 
# 

# In[ ]:




