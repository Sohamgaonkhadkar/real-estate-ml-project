# Gurgaon Real Estate Analytics and Price Prediction System

**Author:** Soham Mahesh Gaonkhadkar  
**Institution:** Indian Institute of Technology Kharagpur  

---

## Abstract
This repository contains an end-to-end Machine Learning pipeline designed to predict real estate prices and provide customized property recommendations in the Gurgaon housing market. The project transitions from raw, unstructured property data to a robust predictive model. The pipeline emphasizes rigorous data preprocessing, the mitigation of optimization biases, algorithmic categorical handling, and model interpretability using SHAP explanations.

## 1. Data Collection and Predictive Feature Space
The initial phase handles the extraction and standardization of heterogeneous data sources for flats and independent houses (`gurgaon_properties.csv`). 

**The Predictive Metrics (Features):**
To accurately predict the target variable (`price` in Crores), the model processes 12 distinct structural, locational, and qualitative metrics:
* **Numerical Metrics:** `built_up_area` (sq. ft.), `bedRoom`, `bathroom`, `servant room`, and `store room`.
* **Categorical Metrics:** `sector` (geographic location), `property_type` (flat vs. house), `agePossession` (property age), `balcony`, `furnishing_type`, `floor_category`, and `luxury_category` (a composite score representing premium amenities).

**Preprocessing highlights include:**
* Identification and treatment of extreme outliers using statistical methods (e.g., capping/removing unrealistic built-up areas).
* Multivariate missing value imputation and logical mapping of ordinal/categorical variables.

## 2. Exploratory Data Analysis (EDA) & Insights
A deep dive into the underlying data distribution was conducted using univariate and multivariate analysis to extract actionable market insights.

**Key EDA Insights:**
* **Price Distribution:** The target variable (`price`) exhibited a highly right-skewed log-normal distribution. To stabilize variance and reduce skewness, a log transformation (`log1p`) was applied.
* **Property Type vs. Metrics:** Multivariate analysis indicated clear demarcations between independent houses and flats. Houses consistently exhibited higher variance in `price`, `built_up_area`, and `price_per_sqft` compared to flats.

## 3. Advanced Categorical Engineering: The Native XGBoost Engine
One of the primary challenges in this dataset was the high cardinality of the `sector` variable (over 100 distinct sectors). 

Standard encoding techniques proved inefficient:
* **One-Hot Encoding** creates a sparse high-dimensional feature space, increasing model complexity.
* **Target Encoding** was initially tested but was found to over-smooth locational variance and introduce subtle target leakage during cross-validation hyperparameter tuning.

**The Selected Strategy:** I abandoned external encoders and utilized **XGBoost’s Native Categorical Support using Histogram-based Trees** (`enable_categorical=True` with `tree_method='hist'`). Rather than reducing a complex geographic sector to a single flat probability, this algorithm keeps the categories intact, allowing the trees to learn non-linear partitions across sector groups and property characteristics.

## 4. Model Selection and Rigorous Evaluation
The modeling phase compared multiple algorithms to establish an optimal bias-variance tradeoff. **Models were rigorously evaluated using a 3-way Train/Validation/Test split and K-Fold cross-validation to ensure out-of-sample generalizability.** *(Note: The target variable 'price' is measured in Crores. The Mean Absolute Error (MAE) below reflects the absolute error in Crores after reverting the log-transformation).*

### Baseline Model Performance (Tree Ensembles)

| Model | R-Squared (R2) | Mean Absolute Error (MAE) | Equivalent Error |
| :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **0.9047** | **0.4475 Cr** | **₹44.7 Lakhs** |
| Random Forest | 0.9008 | 0.4528 Cr | ₹45.2 Lakhs |
| Extra Trees | 0.9010 | 0.4560 Cr | ₹45.6 Lakhs |
| Gradient Boosting | 0.8893 | 0.5100 Cr | ₹51.0 Lakhs |
| Linear Regression | 0.8295 | 0.7130 Cr | ₹71.3 Lakhs |

### Overcoming Optimization Bias (Optuna)
During Bayesian Optimization using **Optuna**, two critical adjustments were made to ensure the model generalized more reliably to unseen real-world data without overfitting:
1. **The Log-Space Optimization Trap:** When optimizing on a log-transformed target, models naturally prioritize minimizing percentage errors rather than absolute value errors, under-emphasizing large absolute errors on high-priced luxury properties. This was fixed by creating a **Custom Scorer** that mathematically forced Optuna to calculate cross-validation MAE in the *original Crores scale*.
2. **Early Stopping:** The optimal number of trees (`best_iteration`) was dynamically found using the completely isolated Validation Set to strictly prevent the model from overfitting to the training distribution. 

**Final Results:** The optimized, early-stopped XGBoost model minimized the error profile, achieving a final out-of-sample **Test MAE of 0.4266 Cr (₹42.66 Lakhs)** and a **Test R² of 0.8751** (Optimized parameters: `learning_rate=0.0269`, `max_depth=7`, `subsample=0.885`, `colsample_bytree=0.701`). 

## 5. Model Explainability & Feature Importance (SHAP)
To ensure the XGBoost model did not function as a pure "black box," **SHAP (SHapley Additive exPlanations)** was integrated into the pipeline to extract global and local feature importance.

* **Methodology:** A custom prediction wrapper was built to feed the categorical data through `shap.Explainer`. Interpretability was divided into two tiers:
  * **Global Interpretability (Beeswarm Plot):** Used to identify macroeconomic drivers across the entire dataset (e.g., proving `built_up_area` is the strongest universal predictor).
  * **Local Interpretability (Waterfall Plots):** Deployed to deconstruct individual predictions, allowing business users to see the exact monetary penalty or premium applied to a specific property based on its unique sector or luxury score.

## 6. Advanced Recommender System (Weighted Ensemble Approach)
A major component of this project is the Content-Based Recommender System (`recommender-system.ipynb`). Rather than relying on a single similarity metric, the system computes and fuses multiple contextual vectors to capture multiple aspects of property similarity.

**Methodology:**
1. **Amenities Vectorization (TF-IDF):** Extracted property facilities (e.g., swimming pool, gym) and vectorized them using `TfidfVectorizer` to compute the first cosine similarity matrix (`cosine_sim1`).
2. **Structural & Categorical Data:** Normalized numerical features and categorical data to form the second similarity matrix (`cosine_sim2`).
3. **Spatial & Location Advantages:** Converted text-based nearby distances (e.g., "2.5 Km from Hospital") into standardized numerical meters to compute a geographic similarity matrix (`cosine_sim3`).

**The Weighted Ensemble Similarity Approach:**
These individual matrices were combined into a master matrix using a weighted formula:  
`cosine_sim_matrix = 30 * cosine_sim1 + 20 * cosine_sim2 + 8 * cosine_sim3`

*Note: The weights were empirically selected through iterative experimentation to balance amenities, structural similarity, and locational influence.* This specific weighting mechanism results in more context-aware and balanced property recommendations.

## 7. Limitations
To maintain scientific rigor, the following system limitations are acknowledged:
* **Data Dependency:** Model performance is inherently dependent on baseline data quality and listing consistency.
* **Macro Factors:** External economic trends, interest rate fluctuations, and temporal market shifts were not incorporated into this static dataset.
* **Heuristic Weighting:** The recommendation engine weights were empirically selected and may require further calibration based on real-world A/B testing and user feedback.

## 8. Repository Structure
* **`data-preprocessing-*.ipynb`**: Scripts for initial cleaning, merging, and formatting.
* **`eda-*.ipynb`**: Notebooks containing the univariate and multivariate statistical analysis.
* **`feature-*.ipynb`**: Dimensionality reduction, derived metric creation, and collinearity dropping.
* **`model-selection.ipynb`**: The core ML pipeline featuring XGBoost Native Categorical Encoding, Custom Scorer metrics, Optuna optimization, and SHAP.
* **`recommender-system.ipynb`**: The weighted-ensemble content-based recommendation engine.

## 9. Technical Stack
* **Language:** Python 3.x
* **Core Libraries:** Pandas, NumPy, Scikit-Learn
* **Modeling & Optimization:** XGBoost (Native Categorical Support using Histogram-based Trees), Optuna
* **Model Interpretability:** SHAP (SHapley Additive exPlanations)
* **NLP & Similarity:** TF-IDF, Cosine Similarity
