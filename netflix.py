
from google.colab import drive
drive.mount('/content/gdrive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from scipy import sparse
import dask.dataframe as dd
import dask.bag as db
from dask.delayed import delayed
dir_path = '/content/gdrive/My Drive/netflix/'
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
!pip install --upgrade surprise
from surprise import Reader, Dataset
import xgboost as xgb
from surprise import BaselineOnly, KNNBaseline , SVD, SVDpp

start_time = datetime.now()
if not os.path.isfile(dir_path + 'netflix_data.csv'):
    data = open(dir_path + 'netflix_data.csv', mode='w')
    row = []
    files = [
            dir_path + 'combined_data_1.txt',
            dir_path + 'combined_data_2.txt',
            dir_path + 'combined_data_3.txt',
            dir_path + 'combined_data_4.txt'
        ]
    for file in files:
        print("Reading the file {}...".format(file))
        with open(file) as f:
            for line in f:
                del row[:]
                line = line.strip()
                if line.endswith(':'):
                    movie_id = line.replace(':','')
                else:
                    row = [x for x in line.split(',')]
                    row.insert(0,movie_id)
                    data.write(','.join(row))
                    data.write('\n')
        print("Done...")
    data.close()
print("Time Taken =" ,datetime.now()-start_time )

df = pd.read_csv(dir_path + 'netflix_data.csv')
df.shape

# Only run this block when you dont have sorted data by 'time'.
if not os.path.isfile(dir_path + "netflix_sorted_data"):
    start_time = datetime.now()

    # creating the dataframe with 4 columns; namely 'movies_id','customer','rating','date'.
    print("Creating the dataframe")
    df = pd.read_csv(dir_path + "netflix_data.csv", sep=',', names=['movies_id','customer','rating','date'])

    # Checking the null value and then delete them if any.
    print("Checking Null values")
    print("Number of nan values: ",sum(df.isnull().any()))
    df = df.dropna(axis=0, inplace= False)

    # Checking the duplicate entries and remove them.
    dups_bool = df.duplicated(['movies_id','customer','rating'])
    dups = sum(dups_bool)
    print("Total number of duplicate entries: {}".format(dups))
    df = df.drop_duplicates()

    # Sorting the dataframe by date column.
    print("Sorting the dataframe")
    df = df.sort_values(by='date', inplace=False)
    print("Done")
    print("Saving... Please wait")
    df.to_csv(dir_path + "netflix_sorted_data", index=False)
    print("Saved")
    print("Time Taken: ", datetime.now()-start_time)
else:
    df = pd.read_csv(dir_path + 'netflix_sorted_data', sep=',' )

print("Total number of ratings in whole dataset: ", df.shape[0])
print("Total number of Users in whole dataset: ", len(np.unique(df.customer)))
print("Total number of Movies in whole dataset: ",len(np.unique(df.movies_id)))

df.customer.describe()

df.rating.describe()

"""<h2>Spliting the data into Train, Cross Validate and Test dataset.</h2>"""

#saving the datasets and loading them.
start = datetime.now()
if not os.path.isfile(dir_path + "train.csv"):
  df.iloc[:int(df.shape[0]*0.6)].to_csv(dir_path + "train.csv")

if not os.path.isfile(dir_path + "cv.csv"):
  df.iloc[int(df.shape[0]*0.6):int(df.shape[0]*0.8)].to_csv(dir_path + "cv.csv")

if not os.path.isfile(dir_path + "test.csv"):
  df.iloc[int(df.shape[0]*0.8):].to_csv(dir_path + "test.csv")
print("Time Taken: ",datetime.now()-start)

train_df = pd.read_csv(dir_path + "train.csv", parse_dates=['date'])
cv_df = pd.read_csv(dir_path + "cv.csv", parse_dates=['date'])
test_df = pd.read_csv(dir_path + "test.csv", parse_dates=['date'])
print("Datasets loaded successfully.")

"""<h2> Statistic on dataset"""

# This block tells you no. of datapoints in a dataset ,no. of rating, no. of unique users and movies
print("Train Dataset")
print("-"*50)
print("Number of datapoints: ",train_df.shape[0],"(", np.round((train_df.shape[0]/(train_df.shape[0]+cv_df.shape[0]+test_df.shape[0]))*100),3,"%",")")
print("\nNumber of ratings: {}".format(train_df.rating.shape[0]))
print("Number of users: {}".format(len(np.unique(train_df.customer))))
print("Number of movies: {}".format(len(np.unique(train_df.movies_id))))
print("\n")
print("Cross Validate Dataset")
print("-"*50)
print("Number of datapoints: ",cv_df.shape[0],"(", np.round((cv_df.shape[0]/(train_df.shape[0]+cv_df.shape[0]+test_df.shape[0]))*100),3,"%",")")
print("\nNumber of ratings: {}".format(cv_df.rating.shape[0]))
print("Number of users: {}".format(len(np.unique(cv_df.customer))))
print("Number of movies: {}".format(len(np.unique(cv_df.movies_id))))
print("\n")
print("Test Dataset")
print("-"*50)
print("Number of datapoints: ",test_df.shape[0],"(", np.round((test_df.shape[0]/(train_df.shape[0]+cv_df.shape[0]+test_df.shape[0]))*100),3,"%",")")
print("\nNumber of ratings: {}".format(test_df.rating.shape[0]))
print("Number of users: {}".format(len(np.unique(test_df.customer))))
print("Number of movies: {}".format(len(np.unique(test_df.movies_id))))

def human(num , units = 'M'):
  units = units.lower()
  num = float(num)
  if units == 'k':
    return str(num/10**3) + 'K'
  if units == 'm':
    return str(num/10**6) + 'M'
  if units == 'b':
    return str(num/10**9) + 'B'

print("Occurance of each rating in train dataset.\n")
train_df.rating.value_counts()

fig, ax = plt.subplots()
plt.title('Distribution of Ratings in Training Dataset', fontsize=15)
sns.countplot(x= train_df.rating)
ax.set_ylabel('No. of Ratings (Millions)')
ax.set_yticklabels([human(item, 'M') for item in ax.get_yticks()])
plt.grid()
plt.show()

pd.options.mode.chained_assignment = None
train_df['days_of_week'] = train_df.date.dt.day_name()
train_df.tail(10)

ax = train_df.resample('d' , on='date')['rating'].count().plot()
ax.set_yticklabels([human(item,'M') for item in ax.get_yticks()])
ax.set_title("Plot showing ratings per day in train dataset")
plt.xlabel("Day")
plt.ylabel("Number of ratings per day")
plt.grid()
plt.show()

ax = train_df.resample('m' , on='date')['rating'].count().plot()
ax.set_yticklabels([human(item,'M') for item in ax.get_yticks()])
ax.set_title("Plot showing ratings per month in train dataset")
plt.xlabel("Months")
plt.ylabel("Number of ratings per month")
plt.grid()
plt.show()

print("Number of ratings given by top 5 users.")
no_of_rating_per_user = train_df.groupby(by='customer')['rating'].count().sort_values(ascending = False)
no_of_rating_per_user.head()

fig = plt.figure(figsize = plt.figaspect(.5))
ax1 = plt.subplot(1,2,1)
plt.title("PDF (Ratings)")
plt.xlabel("No of users")
plt.ylabel("No.of rating")
sns.kdeplot(no_of_rating_per_user, ax=ax1,shade= True)

ax2= plt.subplot(1,2,2)
plt.title("CDF")
plt.xlabel('users')
plt.ylabel('rating')
sns.kdeplot(no_of_rating_per_user,cumulative = True, ax = ax2, shade = True)
plt.show()

no_of_rating_per_user.describe()

quantiles = no_of_rating_per_user.quantile(np.arange(0,1.01,0.01), interpolation = 'higher')
quantiles.plot()
plt.title("Quantiles and their values")
plt.xlabel("Values of quantile")
plt.ylabel("No of rating given by user")
# quantiles with 0.05 intervals
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], color='red',label= "quantiles with 0.05 intervals")
# quantiles with 0.25 intervals
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25],color='green',label='quantiles with 0.25 intervals')
plt.legend(loc = 'best')

for x,y in zip(quantiles.index[::25], quantiles.values[::25]):
  ''' s="({} , {})".format(x, y) sets the content of the annotation to a formatted string that includes the values of x and y.
      For example, if x is 0.25 and y is 500, the annotation will display "(0.25, 500)".

      xy=(x, y) specifies the coordinates on the plot where the annotation arrow will point to. It corresponds to the x and y values
      obtained from the quantiles array.

      xytext=(x-0.05, y+500) sets the coordinates where the annotation text will be placed. It determines the position of the text
      relative to the annotation point. In this case, x-0.05 shifts the text slightly to the left of the annotation point, and y+500
      moves the text upward.'''
  plt.annotate(text="({},{})".format(x,y), xy=(x,y), xytext=(x-0.08,y+400))

print("Quantiles of ratings from 1 to 100 with interval of 5")
quantiles[::5]

print("No of ratinga at last 5 percentile: ",sum(no_of_rating_per_user>707))

no_of_movies_per_rate = train_df.groupby(by ='movies_id')['rating'].count().sort_values(ascending=False)
fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca() #Get Current Axes
plt.plot(no_of_movies_per_rate.values)
plt.title("Plot showing movies that have higher ratings")
plt.xlabel("Movies")
plt.ylabel("Ratings")
ax.set_xticklabels([])
plt.show()

fig, ax = plt.subplots()
sns.countplot(x='days_of_week', data=train_df, ax=ax)
plt.title('No of ratings on each day...')
plt.ylabel('Total no of ratings')
plt.xlabel('')
ax.set_yticklabels([human(item, 'M') for item in ax.get_yticks()])
plt.show()

avg_week_df = train_df.groupby(by=['days_of_week'])['rating'].mean()
print(" Average ratings")
print("-"*30)
print(avg_week_df)
print("\n")

start = datetime.now()
if os.path.isfile(dir_path + 'train_sparse_matrix.npz'):
    print("It is present in your netflix folder on google drive, getting it from drive....")
    # just get it from the disk instead of computing it
    train_sparse_matrix = sparse.load_npz(dir_path + 'train_sparse_matrix.npz')
    print("DONE..")
else:
    print("We are creating sparse_matrix from the dataframe..")
    # create sparse_matrix and store it for after usage.
    # csr_matrix(data_values, (row_index, col_index), shape_of_matrix)
    # It should be in such a way that, MATRIX[row, col] = data
    train_sparse_matrix = sparse.csr_matrix((train_df.rating.values, (train_df.customer.values,train_df.movie.values)),)

    print('Saving it into disk for furthur usage..')
    # save it into disk
    sparse.save_npz(dir_path + "train_sparse_matrix.npz", train_sparse_matrix)
    print('Done..\n')
print('It\'s shape is : (user, movie) : ',train_sparse_matrix.shape)
print(datetime.now() - start)

tr_us,tr_mv = train_sparse_matrix.shape
tr_element = train_sparse_matrix.count_nonzero()

print("Sparsity Of Train matrix : {} % ".format(  (1-(tr_element/(tr_us*tr_mv))) * 100) )

start = datetime.now()
if os.path.isfile(dir_path + 'cv_sparse_matrix.npz'):
    print("It is present in your netflix folder on google drive, getting it from drive....")
    # just get it from the disk instead of computing it
    cv_sparse_matrix = sparse.load_npz(dir_path + 'cv_sparse_matrix.npz')
    print("DONE..")
else:
    print("We are creating sparse_matrix from the dataframe..")
    # create sparse_matrix and store it for after usage.
    # csr_matrix(data_values, (row_index, col_index), shape_of_matrix)
    # It should be in such a way that, MATRIX[row, col] = data
    cv_sparse_matrix = sparse.csr_matrix((cv_df.rating.values, (cv_df.customer.values,cv_df.movie.values)),)

    print('Saving it into disk for furthur usage..')
    # save it into disk
    sparse.save_npz(dir_path + "cv_sparse_matrix.npz", cv_sparse_matrix)
    print('Done..\n')
print("It\'s shape is : (user, movie) : ",cv_sparse_matrix.shape)
print(datetime.now() - start)

cv_us,cv_mv = cv_sparse_matrix.shape
cv_element = cv_sparse_matrix.count_nonzero()

print("Sparsity Of Train matrix : {} % ".format(  (1-(tr_element/(cv_us*cv_mv))) * 100) )

start = datetime.now()
if os.path.isfile(dir_path + 'test_sparse_matrix.npz'):
    print("It is present in your netflix folder on google drive, getting it from drive....")
    # just get it from the disk instead of computing it
    test_sparse_matrix = sparse.load_npz(dir_path + 'test_sparse_matrix.npz')
    print("DONE..")
else:
    print("We are creating sparse_matrix from the dataframe..")
    # create sparse_matrix and store it for after usage.
    # csr_matrix(data_values, (row_index, col_index), shape_of_matrix)
    # It should be in such a way that, MATRIX[row, col] = data
    test_sparse_matrix = sparse.csr_matrix((test_df.rating.values, (test_df.customer.values,test_df.movie.values)),)

    print('Saving it into disk for furthur usage..')
    # save it into disk
    sparse.save_npz(dir_path + "test_sparse_matrix.npz", test_sparse_matrix)
    print('Done..\n')
print('It\'s shape is : (user, movie) : ',test_sparse_matrix.shape)
print(datetime.now() - start)

te_us,te_mv = test_sparse_matrix.shape
te_element = test_sparse_matrix.count_nonzero()

print("Sparsity Of Train matrix : {} % ".format(  (1-(te_element/(te_us*te_mv))) * 100) )

def get_average_ratings(sparse_matrix , of_user):
  # chose ax=1 if of_user is True, 0 other wise.
  ax=1 if of_user else 0
  # A1 is used to convert the column matrix into 1-d array.
  # Represents the sum of ratings for that user or movie,
  sum_of_ratings = sparse_matrix.sum(axis=ax).A1
  # Create the boolean matrix denoting whether the user rated the movie or not.
  is_rated = sparse_matrix != 0
  # Represents the number of ratings received by that user or movie.
  no_of_ratings = is_rated.sum(axis=ax).A1
  u, m = sparse_matrix.shape
  # This line of code creates a dictionary where the keys are user or movie IDs, and the values are the corresponding average ratings.
  # The comprehension filters out IDs with zero ratings and calculates the average rating for the remaining IDs by dividing the
  # sum of ratings by the number of ratings.
  average_rating = {i: sum_of_ratings[i] / no_of_ratings[i] for i in range(u if of_user else m) if(no_of_ratings[i] != 0)}
  return average_rating

"""Global rating for train dataset"""

train_average = dict()
train_global_rating = train_sparse_matrix.sum()/ train_sparse_matrix.count_nonzero()
train_average['global_rating'] = train_global_rating
train_average

"""User rating for train dataset"""

user_id = 29
train_average['user_rating'] = get_average_ratings(train_sparse_matrix, of_user = True)
if user_id in train_average['user_rating']:
  print("Average rating of User: ",user_id," is ",train_average['user_rating'][user_id])
else:
  print("There is no rating with user_id {}".format(user_id))

user_id = 10
# train_average['user_rating'] = get_average_ratings(train_sparse_matrix, of_user = True)
if user_id in train_average['user_rating']:
  print("Average rating of User: ",user_id," is ",train_average['user_rating'][user_id])
else:
  print("There is no rating with user_id {}".format(user_id))

"""Movie rating for train dataset"""

movie_id = 29
train_average['movie_rating'] = get_average_ratings(train_sparse_matrix, of_user = False)
if movie_id in train_average['movie_rating']:
  print("Average rating of Movie: ",movie_id," is ",train_average['movie_rating'][movie_id])
else:
  print("There is no rating with movie_id {}".format(movie_id))

"""Observation: By this we observe that the movie with id 29 has an not superhit but average."""

movie_id = 10
train_average['movie_rating'] = get_average_ratings(train_sparse_matrix, of_user = False)
if movie_id in train_average['movie_rating']:
  print("Average rating of Movie: ",movie_id," is ",train_average['movie_rating'][movie_id])
else:
  print("There is no rating with movie_id {}".format(movie_id))

start_time = datetime.now()
figure , (ax1,ax2) = plt.subplots(nrows = 1, ncols=2, figsize= plt.figaspect(.3))
figure.suptitle("Average rating per user and per movie", fontsize=10)
ax1.set_title("User average ratings")
user_average = [ratio for ratio in train_average['user_rating'].values()]
sns.kdeplot(user_average , ax = ax1 , label= 'PDF' )
sns.kdeplot(user_average , cumulative=True , ax= ax1, label= 'CDF')

ax2.set_title("Movie average ratings")
movie_average = [ratio for ratio in train_average['movie_rating'].values()]
sns.kdeplot(movie_average , ax = ax2 , label= 'PDF' )
sns.kdeplot(movie_average , cumulative=True , ax= ax2, label= 'CDF')

ax1.legend()
ax2.legend()

print("Time taken to plot Pdf's and Cdf's per user and per movie: ",datetime.now() - start_time\
      )

"""#Cold Start Problem

## Cold start problem with users
"""

total_user_r = len(np.unique(df['customer']))
user_r = len(train_average['user_rating'])
print("Total number of Users in dataset: ",total_user_r)
print("Number of users present in train dataset: ", user_r)
print("Number of users NOT present in train dataset: ",total_user_r - user_r , ((total_user_r-user_r)/total_user_r)*100, "%")

"""## Cold start problem with movies"""

total_movie_r = len(np.unique(df['movies_id']))
movie_r = len(train_average['movie_rating'])
print("Total number of Movie in dataset: ",total_movie_r)
print("Number of movie present in train dataset: ", movie_r)
print("Number of movie NOT present in train dataset: ",total_movie_r - movie_r , ((total_movie_r-movie_r)/total_user_r)*100, "%")

def compute_user_similarity(sparse_matrix, verbose=False, compute_for_few=False, top=100,verbose_for_n_rows=20,draw_time_taken=True):
  no_of_user, _ = sparse_matrix.shape
  row_ind, col_ind = sparse_matrix.nonzero()
  row_ind = sorted(set(row_ind))
  time_taken = list()
  rows, cols, data = list(), list(), list()
  if verbose: print("Computing top {} similar users".format(top))

  start_time = datetime.now()
  temp = 0
  for row in row_ind[:top] if compute_for_few else row_ind:
    temp = temp+1
    prev = datetime.now()

    sim = cosine_similarity(sparse_matrix.getrow(row),sparse_matrix).ravel()  # (X)^T.X
    top_sim_ind = sim.argsort()[-top:]
    top_sim_val = sim[top_sim_ind]

    rows.extend([row]*top)
    cols.extend(top_sim_ind)
    data.extend(top_sim_val)
    time_taken.append(datetime.now().timestamp() - prev.timestamp())
    # The purpose of this condition is typically to provide periodic progress updates or print intermediate results during a long computation.
    if verbose:
      if temp % verbose_for_n_rows == 0 :
        print("Computing done for {} users [time elapsed : {} ]".format(temp, datetime.now() -start_time))
  if verbose: print("Creating sparse matrix from the computes similarities.")

  if draw_time_taken:
      plt.plot(time_taken, label= 'Time taken for each user')
      plt.plot(np.cumsum(time_taken), label = 'Total time taken')
      plt.legend(loc = 'best')
      plt.grid()
      plt.xlabel("Users")
      plt.ylabel("Time taken in seconds")
      plt.show()
  return sparse.csr_matrix((data,(rows,cols)) , shape=(no_of_user , no_of_user)), time_taken

start = datetime.now()
u_u_sim, _ =compute_user_similarity(train_sparse_matrix , compute_for_few=True, verbose = True, top=100)
print("*"*100)
print("Time Taken : {}".format(datetime.now()-start))

start = datetime.now()
svd = TruncatedSVD(n_components= 100, algorithm='randomized', random_state=15)
trunSvd = svd.fit_transform(train_sparse_matrix)
print("Time Taken: ", datetime.now()-start)

"""Overall, the below code appears to plot the variance explained (expl_var) on the first subplot and the gain in variance explained with one additional latent factor (change_in_expl_var) on the second subplot. Additionally, it adds annotations to specific points on the first subplot to provide additional information about the (latentfactors, expl_var) values at those points.





"""

# An attribute that represents the ratio of the explained variance for each latent factor or component obtained from the SVD.
# It returns an array-like object containing the explained variance ratios in descending order.
exp_var = np.cumsum(svd.explained_variance_ratio_)

# This line creates a figure and two subplots arranged horizontally. ax1 and ax2 are the axes objects representing the two subplots.
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize= plt.figaspect(.5))
ax1.set_ylabel("Varience Explained")
ax1.set_xlabel("Latent Factors")
ax1.plot(exp_var)
# Defines a list of indices to annotate on the plot.
index = [1,2,5,10,25,50,65,80,90,100]
#  Plots scatter points on the first subplot at the specified indices, with a custom color.
ax1.scatter(x= [i-1 for i in index] , y= exp_var[[i-1 for i in index]], c='#8c564b' )

# Annotates text on the plot to label specific points. It iterates over the indices in ind and adds an annotation with the corresponding (latentfactors, expl_var) values.
for i in index:
  text = "({}, {})".format(i, np.round(exp_var[i-1],2))
  # xytext = used to add annotations to specific points (Position) xytext = (i+7, exp_var[i-1] - 0.003)) play with these value for adjustment
  ax1.annotate(text = text, xy = (i-1, exp_var[i-1]), xytext = (i+7, exp_var[i-1] - 0.003))

#  Calculates the change in expl_var for each adjacent pair of elements.
change_in_exp_var = [exp_var[1+1] - exp_var[i] for i in range(len(exp_var)-1)]


ax2.plot(change_in_exp_var)
ax2.set_xlabel("Latent Factor")
ax2.set_ylabel("Gain in variance3 explained with one additional LF !!")
ax2.yaxis.set_label_position("right")
plt.show()

"""More specifically, expl_var represents the cumulative sum of the explained variance ratios for each latent factor or component. The explained variance ratio quantifies the proportion of the total variance in the data that is explained by each latent factor. By summing up these ratios cumulatively, expl_var provides insight into the cumulative amount of variance explained as more latent factors or components are considered.

For example, if expl_var has values [0.2, 0.4, 0.6, 0.8, 1.0], it means that the first latent factor explains 20% of the total variance, the second latent factor explains an additional 20% (40% in total), the third factor explains another 20% (60% in total), and so on. The final value of 1.0 indicates that all the variance in the data has been accounted for by the latent factors considered.
"""

for i in index:
  print("For Latent Factor {} , {}% of variance covered.".format(i, np.round(exp_var[i-1]*100, 2)))

start = datetime.now()
trun_matrix = train_sparse_matrix.dot(svd.components_.T)
print("Time taken: {}".format(datetime.now()- start))

type(trun_matrix) , trun_matrix.shape

if os.path.isfile(dir_path + "trun_sparse_matrix.npz"):
  trun_sparse_matrix = sparse.load_npz(dir_path + "trun_sparse_matrix.npz")
else:
  trun_sparse_matrix = sparse.csr_matrix(trun_matrix)
  sparse.save_npz(dir_path + "trun_sparse_matrix.npz", trun_sparse_matrix)

print("The shape of Truncated SpareseMatrix is : ", trun_sparse_matrix.shape)

start = datetime.now()
trun_u_u_sim = compute_user_similarity(trun_sparse_matrix, verbose=True, compute_for_few = True, top=100, verbose_for_n_rows=10)
print("*"*100)
print("TimeTaken: ",datetime.now()-start)

"""Movie-Movie Similarity"""

start = datetime.now()
if not os.path.isfile(dir_path + "m_m_similarity_sparse.npz"):
  print("File not forund...")
  for i in range(4):
    print(".")
  print("Wait... We are computing file for you")
  m_m_similarity_sparse = cosine_similarity(train_sparse_matrix.T, dense_output=False)
  print("Computed Successfully")
  print("Saving the file in googledrive under netflix folder. ")
  sparse.save_npz(dir_path + "m_m_similarity_sparse.npz", m_m_similarity_sparse)
  print("Saved Successfully")
  print("The shape of movie-movie similarity sparse matrix is: ", m_m_similarity_sparse.shape)
  print("Time Taken to compute: ",datetime.now()-start)
else:
  m_m_similarity_sparse = sparse.load_npz(dir_path + "m_m_similarity_sparse.npz")
  print("File loaded successfully")
  print("The shape of movie-movie similarity sparse matrix is: ", m_m_similarity_sparse.shape)
  print("Time Taken to load: ",datetime.now()-start)

"""This line of code calculates the similarity scores for the current movie with all other movies, sorts the movies based on similarity (in descending order), and stores the indices of the most similar movies in the sim_movies array."""

movie_ids = np.unique(m_m_similarity_sparse.nonzero()[1])

start = datetime.now()
top_sim_mvi = dict()
for movie in movie_ids:
  # .toarray() converts the selected row to a dense array format. This step is necessary because the subsequent operations require an array rather than a sparse matrix.
  # .ravel() flattens the 2D array to a 1D array, ensuring that the similarity scores are in a one-dimensional format.
  # .argsort() returns the indices that would sort the similarity scores in ascending order. Since we want the most similar movies, we need to sort in reverse order.
  # [::-1] reverses the sorted indices, effectively sorting the similarity scores in descending order.
  # [1:] slices the sorted indices starting from index 1, excluding the first element. This is done to exclude the current movie itself from the list of similar movies.
  similar_movies = m_m_similarity_sparse[movie].toarray().ravel().argsort()[: :-1][1:]
  top_sim_mvi[movie] = similar_movies[:100]
print("Time taken: ",datetime.now()-start)
# checking the top 100 similar movies for movie_id 29
top_sim_mvi[29]

"""Reading the Movies titles"""

if os.path.isfile(dir_path + "movie_titles.csv"):
  print("Reading the data from google drive")
  movies_title = pd.read_csv(dir_path + "movie_titles.csv" , sep=',' , header=None, verbose=True,
                             names=['Movie_id', 'Movie_release_year', 'Movie_title'], index_col='Movie_id', encoding = 'ISO-8859-1')
else:
  print("movie_titles.csv not found kindly go to kaggle and download it.")
movies_title.head()

movies_title[movies_title['Movie_title'] == 'Titanic']

mvi_id = 323
print("Movie : {}\n".format(movies_title.loc[323].values[1]))
print("The movie have {} ratings from users.\n".format(train_sparse_matrix[:,mvi_id].getnnz()))
print("There are total {} movies that are similar with '{}' movie but we will see the top fews similar movies.".format(m_m_similarity_sparse[:,mvi_id].getnnz() , movies_title.loc[mvi_id].values[1]))

similarities = m_m_similarity_sparse[mvi_id].toarray().ravel()
similar_index = similarities.argsort()[: : -1] [1:]
similarities[similar_index]
sim_indices = similarities.argsort()[ : : -1][1:]

mvi_id = 323
plt.plot(similarities[sim_indices], label="All Similar movies", linestyle='solid')
plt.plot(similarities[sim_indices[:100]], label="Top 100 movies", color='red', linestyle='dashed')
plt.xlabel("No. of movies")
plt.ylabel("Cosine Similarity")
plt.title("Similar movies for movie: '{}'".format(movies_title.loc[mvi_id].values[1]))
plt.legend()
plt.show()

mvi_id = 323
plt.plot(similarities[sim_indices], label="All Similar movies", linestyle='-', marker='o', color='blue')
plt.plot(similarities[sim_indices[:100]], label="Top 100 movies", linestyle='--', marker='s', color='red')
plt.xlabel("No. of movies")
plt.ylabel("Cosine Similarity")
plt.title("Similar movies for movie: '{}'".format(movies_title.loc[mvi_id].values[1]))
plt.legend()
plt.grid()
plt.show()

"""Thig plot will show that the top 100 movies for movie title
 "Modern Vampires" have 23% to 13% similarities.
"""

print("Top 10 movies similar to ' Modern Vampires'")
movies_title.loc[sim_indices[:10]]

def get_sample_sparse_matrix(sparse_matrix, no_user, no_movie, path ,verbose=True):
  row_index, col_index, rating = sparse.find(sparse_matrix)
  users = np.unique(row_index)
  movies = np.unique(col_index)
  print("Shape of matrix before sampling: ({} X {})".format(len(users) , len(movies)))
  print("Ratings before sampling: ",len(rating))

  np.random.seed(20)
  sample_users = np.random.choice(users, no_user, replace=False)
  sample_movies= np.random.choice(movies, no_movie , replace=False)
  mask = np.logical_and(np.isin(row_index , sample_users), np.isin(col_index, sample_movies))

  sample_sparse_matrix = sparse.csr_matrix((rating[mask] , (row_index[mask] , col_index[mask])), shape= (max(sample_users)+1, max(sample_movies)+1))

  if verbose:
    print("Shape of matrix after sampling: ({} X {})".format(len(sample_users), len(sample_movies)))
    print("Ratings after sampling: ",rating[mask].shape[0])

  print("Saving the matrix into drive.")
  sparse.save_npz(path , sample_sparse_matrix)
  if verbose:
    print("Save successfully")

  return sample_sparse_matrix

path = dir_path + "train_sample_sparse_matrix.npz"
if not os.path.isfile(path):
  print("Creating the Train sample sparse matrix. \n\n Please Wait")
  train_sample_sparse_matrix = get_sample_sparse_matrix(train_sparse_matrix, no_user = 10000, no_movie=1000, path=path)
else:
  print("Loading the Sample Matrix. \n\n Please Wait")
  train_sample_sparse_matrix = sparse.load_npz(path)

path = dir_path + "cv_sample_sparse_matrix.npz"
if not os.path.isfile(path):
  print("Creating the CV sample sparse matrix. \n\n Please Wait")
  cv_sample_sparse_matrix = get_sample_sparse_matrix(cv_sparse_matrix, no_user = 5000, no_movie=500, path=path)
else:
  print("Loading the Sample Matrix. \n\n Please Wait")
  cv_sample_sparse_matrix = sparse.load_npz(path)

path = dir_path + "test_sample_sparse_matrix.npz"
if not os.path.isfile(path):
  print("Creating the Test sample sparse matrix. \n\n Please Wait")
  test_sample_sparse_matrix = get_sample_sparse_matrix(test_sparse_matrix, no_user = 5000, no_movie=500, path=path)
else:
  print("Loading the Sample Matrix. \n\n Please Wait")
  test_sample_sparse_matrix = sparse.load_npz(path)

"""Featurizing"""

sample_train_dataset = dict()
average_global_rating = train_sample_sparse_matrix.sum()  / train_sample_sparse_matrix.count_nonzero()
sample_train_dataset['global'] = average_global_rating
sample_train_dataset

user = 311465
sample_train_dataset['user'] = get_average_ratings(train_sample_sparse_matrix, of_user=True)
print("Average rating of user {} is: {}".format(user,sample_train_dataset['user'][user]))

'''Movie_release_year    1964.0
   Movie_title           Marnie
   Name: 17109, dtype: object'''
movieid= 17109
sample_train_dataset['movie'] = get_average_ratings(train_sample_sparse_matrix, of_user=False)
print("Average rating for movie '{}' is: {}".format(movies_title.iloc[17109 -1].values[1] ,sample_train_dataset['movie'][movieid] ))

train_sample_user, train_sample_movie,train_sample_rating = sparse.find(train_sparse_matrix)

#user_sim = cosine_similarity(sample_train_sparse_matrix[user], sample_train_sparse_matrix).ravel():
#This line calculates the cosine similarity between the user (represented by the index user) and all other users in the
#sample_train_sparse_matrix. The result is flattened using ravel() to create a 1D array.

# top_sim_users = user_sim.argsort()[::-1][1:]: Here, argsort() sorts the similarity scores in ascending order and returns the indices
 # that would sort the array. By using [::-1], it reverses the order, resulting in indices that sort the array in descending order. [1:]
 # is used to exclude the first element, which corresponds to the user itself. Thus, top_sim_users contains the indices of users that are most similar to the given user.

#top_ratings = sample_train_sparse_matrix[top_sim_users, movie].toarray().ravel(): This line retrieves the ratings of the most similar users
 #for the given movie. It uses the top_sim_users indices and the movie index to access the corresponding ratings from the
 #sample_train_sparse_matrix. The toarray() method converts the sparse matrix slice to a dense array, and ravel() flattens it to a 1D array.

#top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5]): Here, top_ratings[top_ratings != 0] filters out any zero ratings from
#the top_ratings array, and [:5] selects at most the first five non-zero ratings. This ensures that we have a maximum of five ratings from
#similar users for the movie.

#top_sim_users_ratings.extend([sample_train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings))): This line extends the
#top_sim_users_ratings list by appending the average rating of the movie (sample_train_averages['movie'][movie])
#repeated 5 - len(top_sim_users_ratings) times. It ensures that if there are fewer than five ratings from similar users, the remaining slots
#are filled with the average rating.
start = datetime.now()
if os.path.isfile(dir_path + "train_regression_data.csv"):
  print("Train Regression Dataset found in the google drive.\nReading...")
  train_regression_data = pd.read_csv(dir_path + "train_regression_data.csv")
  print("Load Successfully and time taken: {}".format(datetime.now()-start))
else:
  print("Preparing {} tuples\nKindly wait...".format(len(train_sample_rating)))
  with open(dir_path + "train_regression_data.csv" , mode = 'w') as train_regression_data_file:
    count = 0

    for user , movie, rating in zip(train_sample_user, train_sample_movie, train_sample_rating):

      st = datetime.now()
      # Finding 5 most similar users.
      user_similarity = cosine_similarity(train_sample_sparse_matrix[user],train_sample_sparse_matrix).ravel()
      top_sim_user = user_similarity.argsort()[::-1][1:]
      # top_sim_user:  (2648546,)
      top_rating   = train_sample_sparse_matrix[top_sim_user , movie].toarray().ravel()
      top_sim_user_rating = list(top_rating[top_rating != 0][:5])
      top_sim_user_rating.extend([sample_train_dataset['movie'][movie]]* (5- len(top_sim_user_rating)))

      # Finding 5 most similar movies.
      movie_similarity = cosine_similarity(train_sample_sparse_matrix[:,movie].T,train_sample_sparse_matrix.T).ravel()
      top_sim_movies = movie_similarity.argsort()[::-1][1:]
      # top_sim_movies:  (17749,)
      top_rating  = train_sample_sparse_matrix[ user ,top_sim_movies].toarray().ravel()
      top_sim_movie_rating = list(top_rating[top_rating != 0][:5])
      top_sim_movie_rating.extend([sample_train_dataset['user'][user]]* (5-len(top_sim_movie_rating)))

      # Preparing the row by inserting values.
      row = []
      row.append(user)
      row.append(movie)
      row.append(sample_train_dataset['global'])
      row.extend(top_sim_user_rating)
      row.extend(top_sim_movie_rating)
      row.append(sample_train_dataset['user'][user])
      row.append(sample_train_dataset['movie'][movie])
      row.append(rating)
      count += 1

      train_regression_data_file.write(','.join(map(str,row)))
      train_regression_data_file.write('\n')

      if count% 10000 == 0:
        print("Done for {} rows and time elapse: {}".format(count,datetime.now()-start))

print("Total time taken: {}".format(datetime.now()-start))


"""<h2>Reading Regression dataset for Train"""

train_regression_data = pd.read_csv(dir_path + "train_regression_data.csv", names=['user','movie','GAvg', 'sur1','sur2','sur3','sur4','sur5',
                                                                               'smr1','smr2','smr3','smr4','smr5','UAvg','MAvg','rating'],header=None)
train_regression_data.head()

"""<h2>Reading Regression dataset for CrossValidate"""

cv_regression_data = pd.read_csv(dir_path + "cv_regression_data.csv", names=['user','movie','GAvg', 'sur1','sur2','sur3','sur4','sur5',
                                                                               'smr1','smr2','smr3','smr4','smr5','UAvg','MAvg','rating'],header=None)
cv_regression_data.head()

"""<h2>Reading Regression dataset for Test"""

test_regression_data = pd.read_csv(dir_path + "test_regression_data.csv", names=['user','movie','GAvg', 'sur1','sur2','sur3','sur4','sur5',
                                                                               'smr1','smr2','smr3','smr4','smr5','UAvg','MAvg','rating'],header=None)
test_regression_data.head()

"""<h2> Overview of Surprise Library</h2>
The Surprise library uses a specific data format called the Surprise Dataset format. This format is designed to facilitate the use of various recommender algorithms provided by the library.

The Surprise Dataset format represents the user-item ratings data in a structured manner that can be easily processed by the library's algorithms. It consists of three main components: users, items, and ratings.

Here's a brief overview of the format:

1. Users: Users are identified by unique user IDs. Each user ID corresponds to a set of ratings given by that user.

2. Items: Items (or items being recommended) are identified by unique item IDs. Each item ID corresponds to a set of ratings received by that item.

3. Ratings: Ratings represent the user-item interactions or preferences. They typically indicate how a user rates or interacts with an item. Ratings can be numerical, binary (e.g., liked/disliked), or explicit/implicit.

The Surprise Dataset format is often constructed using the Dataset class provided by the library. It offers methods like load_from_df, load_from_file, or load_builtin to load data from dataframes, files, or built-in datasets, respectively. These methods convert the data into the Surprise Dataset format.

Once the data is loaded into the Surprise Dataset format, it can be further processed, split into train and test sets, and used as input for training and evaluating recommender algorithms provided by the Surprise library.
"""

!pip install surprise
from surprise import Reader, Dataset
reader    = Reader(rating_scale=(1,5))
dataset   = Dataset.load_from_df(train_regression_data[['user','movie','rating']], reader)
train_set = dataset.build_full_trainset()

cv_set = list(zip(cv_regression_data.user.values , cv_regression_data.movie.values , cv_regression_data.rating.values))
cv_set[:3]

test_set = list(zip(test_regression_data.user.values , test_regression_data.movie.values , test_regression_data.rating.values))
test_set[:3]

"""<h2> Applying Machine Learning Models

These dictionaries are used as the set of RMSE in Train, cv, test sets for all the machine learning model.
"""

train_model_evaluation = {}
cv_model_evaluation = {}
test_model_evaluation = {}
train_model_evaluation, cv_model_evaluation, test_model_evaluation

"""<h2>Utility functions for XGBoost Regression"""

def get_error_matrix(y_true, y_predict):
  rmse = np.sqrt(np.mean([ (y_true[i] - y_predict[i])**2 for i in range(len(y_predict))])) # Root Mean Square Error
  mape = np.mean(np.abs((y_true - y_predict)/y_true)) *100    # Mean Absolute Percentage Error
  return rmse , mape

def run_xgboost(algo, x_train, y_train, x_test, y_test, verbose=True):
  train_result = {}
  test_result  = {}

  print("Training the model")
  start = datetime.now()
  algo.fit(x_train, y_train , eval_metric = 'rmse')
  print("Done\n Time Taken: ",datetime.now()-start)

  print("Evaluating the model on Train data")
  start = datetime.now()
  y_train_predict = algo.predict(x_train)
  train_rmse, train_mape  = get_error_matrix(y_train.values,y_train_predict)

  train_result = { 'rmse': train_rmse ,
                   'mape': train_mape ,
                   'y_predict': y_train_predict}

  print("Evaluating the model on Test data")
  y_test_predict = algo.predict(x_test)
  test_rmse, test_mape = get_error_matrix(y_test, y_test_predict)

  test_result = { 'rmse': test_rmse ,
                  'mape': test_mape ,
                  'y_predict':y_test_predict}


  if verbose:
    print("*"*100)
    print("Train Data")
    print("RMSE: ",train_rmse)
    print("MAPE: ",train_mape)
    print("\n")
    print("*"*100)
    print("Test Data")
    print("RMSE: ",test_rmse)
    print("MAPE: ",test_mape)

    return train_result , test_result

"""<h2>Utility Function for Surprise Model"""

my_seed = 15
# random.seed(my_seed)
np.random.seed(my_seed)

def get_rating(prediction):
  actual = np.array([pred.r_ui for pred in prediction])
  predict = np.array([pred.est for pred in prediction])
  return actual,predict

def get_errors(prediction, print_them=False):
  actual , predict = get_rating(prediction)
  rmse = np.sqrt(np.mean((predict - actual)**2))
  mape = np.mean(np.abs(predict - actual)/actual)*100
  return rmse , mape

def run_surprise(algo, train_set , test_set, verbose=True):
  start = datetime.now()

  train_result = {}
  test_result = {}
  print("Training the surprise model")
  algo.fit(train_set)
  print("Done\n Time Taken: ",datetime.now()-start)

  # Evaluating Train set
  st = datetime.now()
  print("Evaluting the Train set")
  train_predict = algo.test(train_set.build_testset())
  train_actual_rating , train_predict_rating = get_rating(train_predict)
  train_rmse , train_mape = get_errors(train_predict)
  print("Time Taken: ",datetime.now()-st)

  train_result = { 'rmse': train_rmse,
                   'mape': train_mape,
                   'y_predict': train_predict_rating}

  if verbose:
    print("*"*100)
    print("Train Data")
    print("RMSE: ",train_rmse)
    print("MAPE: ",train_mape)

  # Evaluating Test set
  st = datetime.now()
  print("Evaluting the Test set")
  test_predict = algo.test(test_set)
  test_actual_rating , test_predict_rating = get_rating(test_predict)
  test_rmse , test_mape = get_errors(test_predict)
  print("Time Taken: ",datetime.now()-st)

  test_result = { 'rmse': test_rmse,
                   'mape': test_mape,
                   'y_predict': test_predict_rating}

  if verbose:
    print("*"*100)
    print("Test Data")
    print("RMSE: ",test_rmse)
    print("MAPE: ",test_mape)
  print("\n")
  print("~"*100)
  print("Total Time taken to run the algo: ",datetime.now()-start)
  return train_result, test_result

"""<h2> First XGBoost model with 13 handcraft features"""

x_train = train_regression_data.drop(['user', 'movie','rating'], axis = 1)
y_train  = train_regression_data['rating']

x_test = test_regression_data.drop(['user','movie','rating'], axis = 1)
y_test = test_regression_data['rating']

first_xgb = xgb.XGBRegressor(n_job = -1, random_state=15, n_estimator=100, silence= False)
train_result, test_result = run_xgboost(first_xgb, x_train, y_train, x_test, y_test)
train_model_evaluation['first_algo'] = train_result
test_model_evaluation['first_algo'] = test_result
xgb.plot_importance(first_xgb)
plt.show()

"""<h2> Baseline Surprise model"""

bsl_options = {'method':'sgd', 'learning_rate': 0.001}
bsl_algo = BaselineOnly(bsl_options=bsl_options)
bsl_train_result , bsl_test_result = run_surprise(bsl_algo, train_set, test_set, verbose=True)
train_model_evaluation['bsl_algo'] = train_result
test_model_evaluation['bsl_algo'] = test_result

"""<h2> Building model with 13 Features + BaselineOnly"""

train_regression_data['bslpre'] = train_model_evaluation['bsl_algo']['y_predict']
train_regression_data.head(3)

test_regression_data['bslpre'] = test_model_evaluation['bsl_algo']['y_predict']
test_regression_data.head(3)

from xgboost.sklearn import XGBRegressor
x_train = train_regression_data.drop(['user', 'movie', 'rating'],axis=1)
y_train = train_regression_data['rating']

x_test = test_regression_data.drop(['user', 'movie', 'rating'],axis=1)
y_test = test_regression_data['rating']

second_xgb = xgb.XGBRegressor(n_job=-1, n_estimator=100, silence=False, random_state=15)
train_result, test_result = run_xgboost(second_xgb, x_train, y_train, x_test, y_test)

train_model_evaluation['second_xgb_bsl'] = train_result
test_model_evaluation['second_xgb_bsl']  = test_result

xgb.plot_importance(second_xgb)
plt.show()

"""<h2>Surprise KNN with user-user similarity"""

sim_options = { 'user_based':True, 'shrikage':100, "name":'pearson_baseline', 'min_support':2}
bsl_options = { 'method':'sgd'}
knn_baseline = KNNBaseline(sim_options = sim_options, bsl_options=bsl_options, k=20)
knn_u_bsl_train_result , knn_u_bsl_test_result = run_surprise(knn_baseline, train_set, test_set, verbose=True)

train_model_evaluation['knn_bsl_u'] = knn_u_bsl_train_result
test_model_evaluation['knn_bsl_u']  = knn_u_bsl_test_result

"""<h2> Surprise KNN with item-item similarity model"""

sim_options = { 'user_based':False, 'shrikage':100, "name":'pearson_baseline', 'min_support':2}
bsl_options = { 'method':'sgd'}
knn_baseline = KNNBaseline(sim_options = sim_options, bsl_options=bsl_options, k=20)
knn_m_bsl_train_result , knn_m_bsl_test_result = run_surprise(knn_baseline, train_set, test_set, verbose=True)

train_model_evaluation['knn_bsl_m'] = knn_m_bsl_train_result
test_model_evaluation['knn_bsl_m']  = knn_m_bsl_test_result

sim_options = { 'user_based':False,
                "name":'pearson_baseline',
                'min_support':2}
shrinkage = [20,40,60,80,100]
k = [2,4,6,8,10,15,20,30,40]
bsl_options = { 'method':'sgd'}
avg_score = {}
for kval in k:
  for sval in shrinkage:
    print("KNeighbour: ",kval,"shrinkage: ",sval)
    knn_baseline = KNNBaseline(sim_options = sim_options, bsl_options=bsl_options, k=kval, shrinkage=sval)
    knn_m_bsl_train_result , knn_m_bsl_test_result = run_surprise(knn_baseline, train_set, test_set, verbose=True)

train_regression_data['knn_bsl_u'] = train_model_evaluation['knn_bsl_u']['y_predict']
train_regression_data['knn_bsl_m'] = train_model_evaluation['knn_bsl_m']['y_predict']
train_regression_data.head(2)

test_regression_data['knn_bsl_u'] = test_model_evaluation['knn_bsl_u']['y_predict']
test_regression_data['knn_bsl_m'] = test_model_evaluation['knn_bsl_m']['y_predict']
test_regression_data.head(2)

x_train = train_regression_data.drop(['user', 'movie', 'rating'], axis=1)
y_train = train_regression_data['rating']

x_test = test_regression_data.drop(['user', 'movie', 'rating'], axis=1)
y_test = test_regression_data['rating']

third_xgb = xgb.XGBRegressor(n_estimator = 100, n_job = -1, silence= False, randon_state=15)
train_result , test_result = run_xgboost(third_xgb, x_train, y_train, x_test, y_test)

train_model_evaluation['xgb_bsl_knn'] = train_result
test_model_evaluation['xgb_bsl_knn'] = test_result

xgb.plot_importance(third_xgb)
plt.show()

svd = SVD(n_factors = 50, n_epochs = 20, biased= True, random_state= 15, verbose=True)
train_svd_result, test_svd_result = run_surprise(svd, train_set, test_set, verbose=True)

train_model_evaluation['svd'] = train_svd_result
test_model_evaluation['svd']  = test_svd_result

svdpp = SVDpp(n_factors= 50, n_epochs= 20, random_state= 15, verbose=True)
train_svdpp_result, test_svdpp_result = run_surprise(svdpp, train_set, test_set, verbose= True)

train_model_evaluation['svdpp'] = train_svdpp_result
test_model_evaluation['svdpp'] = test_svdpp_result

"""<h2> XGB with 13 features + bslpre + knn_bsl_u + knn_bsl_m + SVD + SVDPP"""

train_regression_data['svd']   = train_model_evaluation['svd']['y_predict']
train_regression_data['svdpp'] = train_model_evaluation['svdpp']['y_predict']
train_regression_data.head(2)

test_regression_data['svd']   = test_model_evaluation['svd']['y_predict']
test_regression_data['svdpp'] = test_model_evaluation['svdpp']['y_predict']
test_regression_data.head(2)

x_train = train_regression_data.drop(['user' , 'movie' , 'rating'], axis=1)
y_train = train_regression_data['rating']

x_test  = test_regression_data.drop(['user', 'movie', 'rating'], axis = 1)
y_test  = test_regression_data['rating']

forth_xgb = xgb.XGBRegressor(n_job=-1, random_state=15)
train_result, test_result = run_xgboost(forth_xgb, x_train, y_train, x_test, y_test)

train_model_evaluation['xgb_bsl_knn_svd_svdpp'] = train_result
test_model_evaluation['xgb_bsl_knn_svd_svdpp']  = test_result

xgb.plot_importance(forth_xgb)
plt.show()

df= pd.DataFrame(train_model_evaluation)
df

df1= pd.DataFrame(test_model_evaluation)
df1
