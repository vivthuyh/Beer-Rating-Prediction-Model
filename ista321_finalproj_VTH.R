# ISTA 321 final project
# Name: Vivian Huynh
# Student ID: 23729664
# Date: 03/06/2025
# Contributions: Since I was working alone, all work done below is independent! 

##### Summary: 
# The goal of this project is to predict the average rating (overall score) of a beer using characteristics such as ABV, number of reviews, and year. 
# To achieve this, I used two different regression models: Multiple Linear Regression (MLR) and Random Forest.
# MLR was used to provide interpretable insights into how each feature impacts beer ratings, while Random Forest was used to capture more complex, non-linear patterns.
# I compared the predictive performance of both models using mean squared error (MSE) to determine which approach better fits the data.
# This combination of models allowed me to assess both the accuracy and interpretability of different modeling techniques applied to beer data.

##### Question: 
# Can we predict the average rating of a beer using its ABV, number of reviews, and year?
# How does a simple linear model (MLR) compare to a more flexible, non-linear model (Random Forest) when predicting beer ratings?
#####

### Libraries Used! 
library(tidyverse)
library(caret)
library(randomForest)

##### Data
# Upload data
beer <- read_csv("https://docs.google.com/spreadsheets/d/1FQvlCVdeGiMttYgBCsUXge3P_KfL8AmI12cvSXxkPuU/gviz/tq?tqx=out:csv")

##### EDA
glimpse(beer)
summary(beer)
unique(beer$style) #over 50 unique values, not sure if its viable to hot encode this? I do think this is probably a good predictor 

hist(beer$number_of_reviews) #extremely right-skewed, there is probably less than 1000 reviews for every beer 

hist(beer$abv) #There is a peak at the bin 3-6 ish (the mode). This means that the majority of beers have an abv of between 3-6. There are a few beers that have a very high abv- 10+ 

hist(beer$year) #left skewed, many of the beer reviews were added in the later years. 

hist(beer$beer_overall_score) # looks kind of normally distributed, there is a peak at 4. Which means most of the ratings are around 4 (out of a score of 5). There are quite a few ratings at 0. Must have been some unhappy drinkers. 

#What I learned: 
# - There are only 4 features that contain numerical data (dbl). I think I want to use one of these as the target feature: number of reviews, abv, year, and beer overall score. Most likely will use the overall score 
#- The summary suggests that there might be outliers in the abv feature (max being 32), I made a histogram and it seems so! This is probably due to the NA values? 
# - Looking at the summary, there are no NA values in any of the features except for abv and year, which has 5932 and 1 respectively. I think these are important columns. So I think I just want to use the median or mean to replace those NA values for the abv  and drop the single row where the NA value is in the year feature. 
# - So far I am thinking about using a multiple linear regression model and a random forest model due to the nature of how this data looks. 

###Preprocessing: 

#I am dropping all of the columns I don't want to use. 
beer<- beer %>%
  select(beer_overall_score, abv, number_of_reviews, year)


#I am imputing the 5932 NA values in the abv column with the median, and dropping the row of data where there is a single NA value in the year column 
beer$abv[is.na(beer$abv)] <- median(beer$abv, na.rm = TRUE)

beer <- beer %>%
  filter(!is.na(year))

#Now I am scaling the data! 
beer <- beer %>%
  mutate(
    abv = scale(abv),
    number_of_reviews = scale(number_of_reviews),
    year = scale(year)
  )

### Modeling: 

#Multiple Linear Regression: 

#Decided to set a seed for reproducibility! 
set.seed(888)

#Fitted the multiple regression model
mlr_model_1 <- lm(beer_overall_score ~ ., data=beer)

summary(mlr_model_1)


split_index <- createDataPartition(beer$beer_overall_score, p=0.8, list=FALSE, times=10)

error_df <- data.frame(matrix(ncol=2, nrow=ncol(split_index)))
colnames(error_df) <- c('test_error', 'fold_number')

for(i in 1:nrow(error_df)){
  train_data <- beer[split_index[,i],]
  test_data <- beer[-split_index[,i],]
  
  mlr_model <- lm(beer_overall_score ~ ., data=train_data)
  
  predictions <- predict(mlr_model, test_data)
  
  mse <- mean((test_data$beer_overall_score - predictions)^2)
  
  error_df[i, 'test_error'] <- mse
  error_df[i, 'fold_number'] <- i 
}

glimpse(error_df)
mlr_avg_mse <- mean(error_df$test_error)
mlr_avg_mse

#Plotted the mse over the fold number for a visual aspect! 
ggplot(error_df, aes(x=fold_number, y=test_error)) +
  geom_line(color='blue') +
  labs(title='Cross Validation MSE per Fold',
       x= 'Fold Number', y='Mean Squared Error')

#Random Forest 

#Fitted a random forest model to the train data we previously split for the prior model
rf_model <- randomForest(beer_overall_score ~ abv + number_of_reviews + year, data = train_data, ntree = 500)

rf_model
importance(rf_model)

rf_predictions <- predict(rf_model,test_data)

rf_mse <- mean((test_data$beer_overall_score - rf_predictions)^2)

#Comparing the two MSEs 
print(paste("Average MLR MSE (10-fold CV):", round(mlr_avg_mse, 4)))
print(paste("Random Forest MSE (single fit):", round(rf_mse, 4)))

##### Conclusion
# This project aimed to predict beer ratings using ABV, number of reviews, and year, and to compare the performance of two models: Multiple Linear Regression (MLR) and Random Forest. 

# Starting with the Multiple Linear Regression model, the 10-fold cross-validated mean squared error (MSE) was 1.1281, which indicates that the model struggled to accurately predict beer ratings. The adjusted R-squared value was only 0.026, meaning the model explained just 2.6% of the variance in beer ratings. This low R-squared suggests that the relationship between the predictors and beer ratings is weak, or that other important features not included in the model (like beer style) play a larger role.

# Despite its lower performance, the MLR model provides valuable insight into how individual features relate to beer ratings. All three predictors (ABV, number of reviews, and year) were statistically significant with p-values well below 0.001, indicating strong evidence that these features influence beer ratings. 

# The coefficient for ABV was 0.1400 (SE = 0.0037), meaning that for every one standard deviation increase in ABV, the predicted beer rating increases by approximately 0.14 points, holding all else constant. This positive relationship makes sense — stronger beers may appeal more to certain drinkers, especially craft beer enthusiasts. The coefficient for number of reviews was 0.0564 (SE = 0.0038), indicating that beers with more reviews tend to have slightly higher ratings, though the effect size is modest. This could reflect the fact that beers with more attention may come from more reputable breweries or have broader appeal. 

# Interestingly, year had a negative coefficient of -0.0810 (SE = 0.0038), meaning that newer beers (higher year values) tend to receive lower ratings, even after controlling for ABV and number of reviews. This might reflect a trend where newer beers have not yet built up the reputation or consistency of older, well-established beers.

# The Random Forest model, on the other hand, achieved a much lower MSE of 0.1538 and explained approximately 86.72% of the variance. This massive improvement highlights Random Forest's ability to capture complex, non-linear relationships and interactions between features — relationships that MLR could not capture due to its linear form. Feature importance from the Random Forest model indicated that the number of reviews was by far the most important predictor, followed by year and ABV.

# Overall, the Random Forest model is the preferred approach for this dataset due to its lower error rate. However, the MLR model remains useful for understanding the individual effects of each predictor. If this project were to be expanded, future steps could include incorporating additional features, such as beer style, region, or brewery characteristics, which might further improve the model performance. I was thinking about doing this myself, but I was unsure how one hot encoding features with over 50 unique values would go. I also think this dataset could be explored with clustering, which could also provide different insights! 
