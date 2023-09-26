# Netflix_movie_recommendation
![Netflix](dataset-cover.jpg)
In today's digital age, online streaming platforms like Netflix have an enormous library of movies and TV shows. However, this vast content library can overwhelm users when it comes to choosing what to watch. To enhance the user experience and keep viewers engaged, Netflix relies heavily on recommendation systems that suggest content tailored to individual preferences.

Netflix is all about connecting people to the movies they love. To help customers find those movies, they developed world-class movie recommendation system: CinematchSM. Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Netflix use those predictions to make personal movie recommendations based on each customer’s unique tastes. And while Cinematch is doing pretty well, it can always be made better.

Now there are a lot of interesting alternative approaches to how Cinematch works that netflix haven’t tried. Some are described in the literature, some aren’t. We’re curious whether any of these can beat Cinematch by making better predictions. Because, frankly, if there is a much better approach it could make a big difference to our customers and our business.

# Key Components:

1. Data Collection: We will start by gathering relevant data, which may include user profiles, movie data (genres, ratings, release dates), and user interaction history (watch history, ratings, reviews).

2. Data Preprocessing: Cleaning and preparing the data is crucial. We will handle missing values, perform feature engineering, and ensure that the data is ready for modeling.

3. Exploratory Data Analysis (EDA): Understand the data distribution, user behavior, and movie preferences through visualization and statistical analysis. EDA can help identify patterns and insights that inform our recommendation strategy.

4. Collaborative Filtering: Implement collaborative filtering techniques, such as user-based or item-based filtering, to find similarities between users or movies. Collaborative filtering leverages the wisdom of the crowd to make recommendations.

5. Content-Based Filtering: Build a content-based recommendation system that considers the characteristics of movies (e.g., genre, director, actors) and matches them with user preferences.

6. Hybrid Recommendation: Combine collaborative filtering and content-based filtering to create a hybrid recommendation system that provides more accurate and diverse recommendations.

7. Model Evaluation: Assess the performance of the recommendation models using appropriate evaluation metrics, such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE).

# Visualization:
1. Barplot: A bar chart is used to display the distribution of ratings in the dataset.
2. Countplot: A count plot is used to display the number of ratings per day and per month.
3. Countplot: Plot showing maximum ratings according to days of week. 
4. Kdeplot: To show the Probability Density Function (PDFs) and Cumulative Distribution Functions (CDFs) of ratings as per users.

# Result/Output:

Input query: 'Modern Vampires'
Recommended movies: Top 10 movies similar to 'Modern Vampires'
Movie_release_year	   Movie_title       Movie_id		
4667                  	1996.0	         Vampirella
15237	                  2001.0         	 The Forsaken
67	                    1997.0	         Vampire Journals
16279	                  2002.0           Vampires: Los Muertos
13873	                  2001.0	         The Breed
4173                  	1998.0         	 From Dusk Till Dawn 2: Texas Blood Money
1900                  	1997.0	         Club Vampire
13962                 	2001.0	         Dracula: The Dark Prince
15867	                  2003.0	         Dracula II: Ascension
3496	                  1998.0	         Vampires
 
# Expected Outcome:

By the end of this project, you will have built a functional Netflix-style movie recommendation system that provides users with tailored movie suggestions based on their watching history and preferences. You will gain hands-on experience in data preprocessing, recommendation algorithms, and model evaluation.

# Skills Demonstrated:

1. Data collection and preprocessing
2. Exploratory data analysis (EDA)
3. Collaborative filtering and content-based filtering
4. Model evaluation and selection
5. Hybrid recommendation system development
