# Group 4 Project 2 Report

### Authors
Pratham Choksi, Emmanuel Leonce, Vicky Singh, Alec Pixton

## Overview

This project explores the factors that influence house prices, particularly those valued above $1 million, using data on homes sold in King County, USA. The project includes data analysis, visualizations, and statistical modeling to identify significant predictors and assess their influence on housing prices.

## Table of Contents
- [Summary](#summary)
- [Variables](#variables)
  - [Existing Variables](#existing-variables)
  - [Created Variables](#created-variables)
- [Questions of Interest](#questions-of-interest)
- [Visualizations](#visualizations)
- [Models](#models)
  - [Question 1 Model](#question-1-model)
  - [Question 2 Model](#question-2-model)
- [Conclusion](#conclusion)

## Summary

The team investigated the variables most influential to housing prices and the likelihood of a house selling for over $1 million.

### Key Findings:
1. **Square Footage**:
   - The square footage of living space (sqft_living) had the strongest correlation (0.7) with price.
   - Homes with larger sqft_living generally have higher prices but show greater variability.

2. **Waterfront and View Quality**:
   - Homes on the waterfront are 4.67 times more likely to sell for over $1 million.
   - Homes with the best views are 10.19 times more likely to sell for over $1 million.

3. **Model Performance**:
   - The area under the curve (AUC) for the logistic regression model is 0.939, indicating excellent predictive performance.

## Variables

### Existing Variables
- **id**: Unique ID for each home sold.
- **date**: Date of the home sale.
- **price**: Price of each home sold.
- **bedrooms**: Number of bedrooms.
- **bathrooms**: Number of bathrooms (0.5 indicates a room with a toilet but no shower).
- **sqft_living**: Square footage of the apartment's interior living space.
- **sqft_lot**: Square footage of the land space.
- **floors**: Number of floors.
- **waterfront**: Dummy variable (1 = on the waterfront, 0 = not on the waterfront).
- **view**: Index (0-4) indicating the quality of the view.
- **condition**: Index (1-5) for the apartment’s condition.
- **grade**: Index (1-13) for construction/design quality.
- **sqft_above**: Square footage of the interior above ground.
- **sqft_basement**: Square footage of the basement.
- **yr_built**: Year the house was built.
- **yr_renovated**: Year of the last renovation.
- **zipcode**: Zipcode of the house.
- **lat**: Latitude.
- **long**: Longitude.
- **sqft_living15**: Square footage of living space for the nearest 15 neighbors.
- **sqft_lot15**: Square footage of land for the nearest 15 neighbors.

### Created Variables
- **above_million**: Binary variable (1 = price > $1,000,000, 0 = otherwise).
- **train/test**: Indicators for training and test dataset splits.
- **grade (mutated)**: Grade categorized as "low" (grade < 8.5) or "high" (grade ≥ 8.5).

## Questions of Interest

### Question 1
What variables are most influential to the price of a house?
- **Response Variable**: Price
- **Predictors**: All relevant quantitative variables in the dataset.

### Question 2
How do waterfront status, view quality, and square footage influence the likelihood of a house being sold for over $1 million?
- **Response Variable**: Binary indicator for sales price above $1 million.
- **Predictors**: Waterfront, view, and sqft_living.

## Visualizations

1. **Correlation Matrix**:
   - Highlighted strong relationships between sqft_living, sqft_above, and price.

2. **Boxplot of Housing Prices by Grade**:
   - Higher grades correlate with higher median prices.

3. **Scatter Plots**:
   - **sqft_living vs. Price**: Showed a positive correlation but greater variability for larger homes.
   - **sqft_above vs. Price**: Similar trends to sqft_living but collinear with sqft_living.

4. **Distribution of View Quality and Prices**:
   - Higher view categories have a greater proportion of homes priced above $1 million.

5. **Waterfront vs. Prices**:
   - Waterfront properties command higher prices with less variability.

## Models

### Question 1 Model
**Model**: Multiple Linear Regression
- Predictors included sqft_living, bedrooms, bathrooms, lat, long, and other quantitative variables.
- **Key Results**:
  - R-squared: 0.628 (62.8% variance explained).
  - Significant predictors: sqft_living, bathrooms, lat, long.
  - Multicollinearity detected (VIF > 5 for sqft_living and sqft_above).

**Improvements**:
- Removed collinear variables (e.g., sqft_above).
- Focused on the most influential predictors.

### Question 2 Model
**Model**: Logistic Regression
- Predictors: Waterfront, view, and sqft_living.
- **Key Results**:
  - Waterfront properties: Odds ratio = 4.67.
  - Best view (view4): Odds ratio = 10.19.
  - sqft_living: Small but significant effect.
  - AUC: 0.939, indicating excellent predictive performance.

**Validation**:
- Confusion Matrix:
  - Accuracy: 95.06%
  - Sensitivity: 42.27%
  - Specificity: 98.89%

## Conclusion

This project demonstrates that square footage, waterfront status, and view quality are critical factors in housing prices. While the models provide meaningful insights, multicollinearity and high variability in housing prices present challenges to precise predictions. Further refinement could include additional data or advanced modeling techniques to improve accuracy.

