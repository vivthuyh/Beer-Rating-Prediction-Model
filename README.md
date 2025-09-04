# Beer-Rating-Prediction-Model
Built Multiple Linear Regression and Random Forest models to predict beer ratings using ABV, reviews, and year. 

# 🍺 Predicting Beer Ratings with Regression Models  

## 📖 Overview  
This project explores whether the **average rating of a beer** can be predicted using just a few features: **ABV (alcohol by volume)**, **number of reviews**, and **year**.  

I compared two modeling approaches:  
- **Multiple Linear Regression (MLR):** interpretable but limited.  
- **Random Forest:** powerful for non-linear patterns.  

The goal was to evaluate both **accuracy** and **interpretability**.  

---

## ❓ Research Questions  
- Can beer ratings be predicted from ABV, number of reviews, and year?  
- How does a simple linear model compare to a flexible, non-linear model?  

---

## ⚙️ Tech Stack  
- **Language:** R  
- **Libraries:**  
  - `tidyverse`  
  - `caret`  
  - `randomForest`  

---

## 📊 Data  
- **Source:** [Beer Dataset (Google Sheets CSV)](https://docs.google.com/spreadsheets/d/1FQvlCVdeGiMttYgBCsUXge3P_KfL8AmI12cvSXxkPuU/gviz/tq?tqx=out:csv)  
- **Features used:**  
  - `abv` (% alcohol by volume)  
  - `number_of_reviews`  
  - `year`  
  - `beer_overall_score` (target)  

**Preprocessing steps:**  
- Imputed missing `abv` values with the median.  
- Dropped one missing `year` value.  
- Scaled all numeric features.  

---

## 🚀 Modeling Approach  

### 🔹 Multiple Linear Regression (MLR)  
- Used **10-fold cross-validation** for evaluation.  
- Interpretable coefficients for each predictor.  

### 🔹 Random Forest  
- Trained with **500 trees**.  
- Captured non-linear relationships.  
- Evaluated on the same train/test split.  

---

## 📈 Results  

| Model                      | Mean Squared Error (MSE) | Variance Explained |
|-----------------------------|--------------------------|---------------------|
| Multiple Linear Regression  | **1.1281** (CV avg)      | ~2.6%              |
| Random Forest               | **0.1538**               | ~86.7%             |

**MLR Insights:**  
- Higher **ABV** → slightly higher ratings.  
- More **reviews** → slightly higher ratings.  
- Later **years** → slightly lower ratings.  

**Random Forest Insights:**  
- Outperformed MLR by a wide margin.  
- **Number of reviews** = most important predictor, followed by year and ABV.  

---

## 🏁 Conclusion  
- **Random Forest** is the best predictive model for this dataset.  
- **MLR** provides interpretability, showing feature-level effects.  
- **Future work:**  
  - Add categorical features (beer style, brewery region).  
  - Explore clustering techniques for deeper insights.  

---

## 📂 How to Run  

1. Clone this repo.  
2. Open `ista321_finalproj_VTH.R` in **RStudio**.  
3. Install required packages:  

```r
install.packages(c("tidyverse", "caret", "randomForest"))
