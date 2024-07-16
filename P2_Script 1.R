## Project 2
## Pratham Choksi, Emmanuel Leonce, Alec Pixton, and Vicky Singh

## Set Up and Load Data

library(tidyverse)
library(ggcorrplot)
library(ggplot2)
library(gridExtra)
library(car)
library(ROCR)

data <- read.table("kc_house_data.csv", header = TRUE, sep = ",", row.names = NULL)
head(data)

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Creating and Converting Variables

# Create Binary Response Variable
data$above_million <- ifelse(data$price > 1000000, 1, 0)

# Convert 'above_million', 'waterfront', and 'view' to factor
data$above_million <- as.factor(data$above_million)
data$waterfront <- as.factor(data$waterfront)
data$view <- as.factor(data$view)

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Train and Test Data Split

set.seed(6021)

sample.data <- sample.int(nrow(data), floor(.50*nrow(data)), replace = F)
train <- data[sample.data, ]
test <- data[-sample.data, ]

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Correlation Matrix

numeric_data <- train[, sapply(train, is.numeric)]

correlation_matrix <- cor(numeric_data)
ggcorrplot(correlation_matrix, hc.order = TRUE, type = "lower", lab = TRUE)

price_correlation <- correlation_matrix["price", ]
print(price_correlation)

# The correlation matrix highlights key relationships between house features and pricing. A prominent correlation of 0.59 between living area size and price suggests 
# larger homes command higher prices. Meanwhile, waterfront and view features mildly correlate with price, indicating a modest impact on value. Strong correlations 
# between 'sqft_above', 'grade', and 'sqft_living' emphasize the importance of living space in property valuation. This matrix is an invaluable tool for understanding 
# the variables that influence house pricing in the real estate market. 

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Visualization 1: Grade vs. Price 

train<-train%>%
  mutate(level=ifelse(grade<8.5,"Low","High"))

library(ggplot2)
ggplot(train, aes(x = level , y = price)) +
  geom_boxplot(fill = 'lightblue') +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 1)) +
  labs(x = 'grade', y = 'pdice', title = "Distribution of housing price by Grade")

# From boxplot of Distribution of housing price by Grade, As housing grade increase, the housing price also increase.

# We have left skewed boxplot distribution of housing grade above 8.5 with median housing price around $750,000 which is higher compare to the left skewed boxplot 
# distribution of housing grade below 8.5, with median housing price around $375000.

# We have outliers in the both left skewed boxplot distribution of housing grade, which indicate unusual observations that may warrant further investigation since 
# they might indicate Measurement errors, Data entry errors, Distribution characteristics (long tails), Sampling issues (the data point actually belongs to another 
# population).


## Visualization 2: Living Square Feet vs. Price 

ggplot(data, aes(x = sqft_living, y = price)) +
  geom_point() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 1)) +
  labs(x = "sqft_living", y = "Price", title = "Distribution of housing price by sqft_living")

# This scatter plot demonstrates the relationship between the size of the living space in square feet and the price of houses. A dense cluster of data points shows 
# that as the living square footage increases, there is a general upward trend in price. However, the spread of data points becomes wider with larger living spaces, 
# indicating a greater variability in price for larger properties. This trend highlights the square footage as a significant factor in housing prices, with larger 
# homes tending to be more expensive, yet with a varied price range as size increases. 


## Visualization 3: square feet above vs. Price 

ggplot(data, aes(x =  sqft_above, y = price)) +
  geom_point() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 1)) +
  labs(x = "sqft_above", y = "Price", title = "Distribution of housing price by sqft_above")

# This scatter plot demonstrates the relationship between the housing square feet above and the price of houses. A dense cluster of data points shows that as the 
# housing square feet above increases, there is a general upward trend in price. However, the spread of data points becomes wider with larger square feet above, 
# indicating a greater variability in price for larger properties. This trend highlights the square feet above as a significant factor in housing prices, with larger 
# homes tending to be more expensive, yet with a varied price range as size increases. 

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Question 2: Visualizations


## Univariate Visualization: view by above_million

ggplot(data, aes(x = view, fill = above_million)) +
  geom_bar() +
  theme(plot.title = element_text(hjust = .5)) +
  labs(x = 'View', y = 'Count of Homes', title = 'Bar Chart of Home View Category', fill = '> $1,000,000')

# This visualization illustrates the distribution of homes based on their "view" categories, with a distinction between homes that sell for over $1 million and those
# that do not. The majority of homes fall into category 0 for View, where most homes are selling below $1 million. As the view quality improves, the proportion of
# homes selling for more than $1 million significantly increases, particularly in categories 3 and 4. This suggests that higher-quality, or more desirable, views may
# correlate with higher sales prices. This highlights a potential premium on properties with better views, which is a valuable insight for real estate pricing
# strategies.


## Bivariate Visualization: price vs. waterfront

ggplot(data, aes(x = waterfront, y = price)) +
  geom_violin() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 1)) +
  labs(x = 'Waterfront', y = 'Price', title = "Violin Plot of Price vs. Waterfront")

# This visualization illustrates the distribution of home prices relative to their waterfront status, with properties that are on the waterfront being designated as 1.
# Generally, non-waterfront gomes have a broader distribution of prices that are typically lower, with a more pronouced peat at the lower price range. In contrast,
# waterfront homes have a much narrower distribution, indicating less variability in price, but significantly higher overall prices with a sharp peak at higher values. 
# This clearly demonstrates the premium attached to waterfront properties, which command higher prices.


## Bivariate Visualization: price vs. sqft_living

ggplot(data, aes(x = sqft_living, y = price)) +
  geom_point() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 1)) +
  labs(x = 'Square Feet (Living)', y = 'Price', title = "Scatter Plot of Square Feet (Living) vs. View")

# This visualization illustrates the relationship between the swuare footage of living space and the prices of homes. The plot reveals a general trend where larger
# living areas tend to correlate with higher house prices. This trend is particularly evident as you move from smaller to larger homes. Additionally, the dispersion
# of price points becomes more varied with an increase in living area, suggesting that factors beyond size, such as location, quality of construction, level of luxury,
# etc., also play a significant role in determining the price.


## Multivariate Visualization: price vs. sqft_living by waterfront and view

ggplot(data, aes(x = sqft_living, y = price)) +
  geom_point(aes(color = view, size = waterfront), alpha = 0.5) +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 1)) +
  labs(x = 'Square Feet (Living)', y = 'Price', title = 'Scatter Plot of Price vs. Square Feet (Living) by Waterfront and View')

# This visualization illustrates a clear trend where, generally, larger living spaces correlate with higher prices. Notably, properties onn the waterfront, as
# indicated by the larger dots, command higher prices across similar living areas compared to non-waterfront properties. This premium for waterfront homes is visible
# across all view categories. Furthermore, homes with higher view ratings, particularly those categorized as 3 and 4, are positioned towards the higher end of the 
# price range, regardless of their living space. This suggests that superior views enhance property value significantly, a trend that is even more pronounced for
# waterfront properties. The concentration of lower-priced gomes largely falls within the view category of 0, indicating that a lack of a significant view tends to
# correspond with more moderate pricing, despite the living space size. This plot effectively highlights the compounded value added by a larger space, superior views,
# and waterfront locations, illustrating how these factors interact to influence real estate pricing.

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Question 1: Linear Regression

library(GGally)
library(readr)
library(caret)

head(data)

filtered_data <- data
filtered_data <- filtered_data[, !(names(filtered_data) %in% c("waterfront", "view", "condition", "grade", "date", "id", "sqft_basement"))]
filtered_data <- na.omit(filtered_data)
#Here I removed any variables are not quantitative or feel is not relevant towards price

model <- lm(price ~ ., data = filtered_data)
summary(model)
#based on the summary of the model, we can see that the r^2 is 0.6278, meaning that
#67.7% of the variance in price can be explained by:

#"price", "bedrooms", "bathrooms"
#"sqft_living", "sqft_lot", "floors", "sqft_above", "yr_built", "yr_renovated"
#"zipcode", "lat", "long", "sqft_living15", "sqft_lot15"

#This model also shows that all the predictors are statistically significant since
#they all have p-values less than 0.05. 
#The t-values provides us with more information on which predictors are most influential.
#It seems like lat, sqft_living, bathrooms, and sqft_living15 seems like the 
#most influential predictors for price. 

#Since the The p-value is less than 2.2e-16, 
#we can conclude that the model is statistically significant at the 0.05 level.


#To test if the model is useful, we performed an ANOVA F test
anova_result <- anova(model)

print(anova_result)
#Our null hypothesis is regression coefficient are equal to 0
#our alternative is that regression coefficients are not equal to 0

#all predictors except floors have p-values less than 0.05, 
#which suggests that all predictors except floors has an effect  
#on price. 

#We can see that majority of the F values except floors is greater than 1, meaning
#that we can reject the null hypothesis for all predictors except floors.

#Since floors was not a great predictors we updated the model to remove floors.
filtered_data <- filtered_data[, !(names(filtered_data) %in% c("floors"))]
model <- lm(price ~ ., data = filtered_data)
summary(model)

#Since this is a multiple linear regression, It is good to check for Multicollinearity
#This will tell us which of the predictors are highly correlated with each other.
#Once quick way to check is to see if the standard error is large for any of the
#predictors. 
model <- lm(price ~ ., data = filtered_data)
summary(model)
#based on the model we can see quite a few standard errors that are large such as
#bedrooms, bathrooms, lat and long. This means we have strong multicollinearity.
#To better look we calculated the VIFs

library(faraway)
faraway::vif(model)
#The larges VIF is with sqft_living with 7.314256 and sqft_above with 5.148665.
#VIFs above 5 indicate a moderate degree of multicollinearity,
#while VIFs above 10 indicate a strong degree of multicollinearity.

#To summarize what we have seen:
#The ANOVA F test is significant, and a lot of the t tests are significant.
#We see huge standard errors for the estimated coefficients.
#The largest VIF is 7.314256

#Collectively, there is high degree of multicollinearity in this model.

#The code below assesses the predictive ability on test data:
X <- subset(filtered_data, select = -c(price))
y <- filtered_data$price

# Split the data into training and testing sets
set.seed(123)  # for reproducibility
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Build the linear regression model
train_data <- cbind(price = y_train, X_train)
modelp <- lm(price ~ ., data = train_data)

# Predict on the testing data
y_pred <- predict(modelp, newdata = X_test)

# Evaluate the model
mse <- mean((y_test - y_pred)^2)
rmse <- sqrt(mse)
print(paste("Mean Squared Error:", rmse))

# Print coefficients to see the influence of each variable
coefficients <- coef(modelp)
print(coefficients)
#Based on the range of price values, We found that the mse was low with all the predictors, so based on
#what we discovered with multicollinearity and p-values, we created a modified
#model to improve the mse with what we found to be the most influential predictors:
#lat, long, sqft_living, sqft_above, bathrooms, and bedrooms

modelp2 <- lm(price ~ sqft_living + sqft_above + bathrooms + bedrooms, data = cbind(train_data))

# Predict on the testing data
y_pred <- predict(modelp2, newdata = X_test)

# Evaluate the model
mse <- mean((y_test - y_pred)^2)
rmse <- sqrt(mse)
print(paste("Mean Squared Error:", rmse))

# Print coefficients to see the influence of each variable
coefficients <- coef(modelp2)
print(coefficients)

#After some experimentation and trial and error we realized that since there is 
#an extremely high multicollinearity, We are able to get a lower mse with all 
#the predictors than just the most influential, however the difference is small.
#the root mse we got with all variables is 233193 while the root mse for the 
#most influential predictors is 243695 to 264416.

#This means that Given that the price variable ranges from 
#$75,000 to $7,700,000, the RMSE value suggests that on average, 
#your model’s predictions are off by about $233,193, Showing the model is 
#moderately accurate.

#In conclusion it is difficult to get a very accurate price predictions since
#there are many predictors that influence each other and have a high influence
#on price. However, we can see that sqft_living + sqft_above + 
#bathrooms + bedrooms seems to cause the most volatility to price changes. 


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Question 2: Logistic Regression


## Logistic Regression Model

logistic_model <- glm(above_million ~ waterfront + view + sqft_living, family = binomial, data = train)

summary(logistic_model)

# Our Logistic Regression Model output, which includes "view," "waterfront," and "sqft_living" as predictors, shows that all included variables are statistically 
# significant predictors of whether a house sells for more than $1 million. With the intercept being approximately -8.37, our model indicates that the baseline
# category (e.g., houses not on the waterfront with the lowest view rating and a smaller sqft_living) has a low probability of exceeding the $1 million sales price
# threshold.

# The Waterfront (waterfront1) coefficient of approximately 1.54 suggests that houses on the waterfront are more likely to sell for more than $1 million, holding
# all other variables constant . The odds ratio can be calculated as exp(1.54), which is 4.67. This indicates that the odds of selling above the designated sales
# price threshold are about 4.67 times higher for waterfront properties than for non-waterfront properties.

# The View coefficients for the different view categories (view1 to view4) generally increase as the view quality improves, suggesting a strong positive relationship
# between the quality of the view and the likelihood of a house selling for more than $1 million. For instance, view4 with a coefficient of approximately 2.32 shows
# a strong positive impact. The odds ratio for view4 can be calculated as exp(2.32), which is 10.19. This is significantly higher than the odds compared to houses
# with the lowest view quality. 

# The square footage of the apartment interior living space (sqft_living) coefficient of 0.0019499 indicates that each additional square foot of living space
# increases the odds of selling a house above $1 million by a factor of 1.002, which seems insignificant but can acculate to a substaintial effect over hundreds
# of square feet.

# This model appears robust and meaningful, especially for practical real estate evaluations where waterfront properties and panoramic views are highly valued.
# Waterfront locations and higher view ratings substantially increase the probability of a house selling for more than $1 million. The square footage of the living
# area also positively impacts the selling price, however the effect per square foot is smaller when compared to the categorical attributes included in the model.


## ROC Curve

preds <- predict(logistic_model, newdata = test, type = 'response')
rates <- ROCR::prediction(preds, test$above_million)
roc_result <- ROCR::performance(rates, measure = 'tpr', x.measure = 'fpr')
plot(roc_result, main = 'ROC Curve for Model')
lines(x = c(0,1), y = c(0,1), col = 'red')

# The Receiver Operating Characteristic (ROC) Curve  is a graphical representation used to assess the performance of binary classification models at various threshold 
# settings. The curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at different threshold levels. Our ROC Curve shows a very good model
# performance as it significantly bows towards the top left corner. This suggests a high TPR and a low FPR across many threshold settings. Essentially, our Logistic
# Regression Model performs well in distinguishing between houses that sell for more than $1 million and those that do not and is better then random guessing.


## Area under the ROC Curve (AUC)

auc <- ROCR::performance(rates, measure = 'auc')
auc@y.values

# A perfect classifier will have an AUC of 1. A classifier that randomly guesses will have an AUC of 0.5. Thus, AUCs closer to 1 are desirable. The AUC of our model 
# is 0.9393754, which means our Logistic Regression Model does better than random guessing.


## Confusion Matrix

conf_matrix <- table(test$above_million, preds > 0.5)
conf_matrix

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix) 
sensitivity <- conf_matrix[2,2] / sum(conf_matrix[2,]) 
specificity <- conf_matrix[1,1] / sum(conf_matrix[1,])

list(Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity)

# According to the Confusion Matrix: 9964 observations were correctly predicted as not selling above $1 million (True Negative), 112 observations were incorrectly 
# predicted as selling above $1 million (False Positive), 422 observations that sold above $1 million were incorrectly predicted as not doing so (False Negative), and 
# 309 observations were correctly predicted as selling above $1 million (True Positive).

# According to the accuracy, our model correctly predicts the outcome approximately 95.06% of the time. The sensitivity of our model states that we identify
# approximately 42.27% of the houses that sell for more than $1 million correctly. The specificity of our model states the proportion of actual negatives that were 
# correctly identified, meaning the model correctly predicts approximately 98.89% of the houses that do not sell for more than $1 million.
