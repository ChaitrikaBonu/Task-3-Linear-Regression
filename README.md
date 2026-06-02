# AI & ML Internship - Task 3

## Linear Regression using Titanic Dataset

### Objective

The objective of this task is to implement and understand Linear Regression using Scikit-learn. The model is trained to predict passenger fare based on selected features from the Titanic dataset.

### Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* KaggleHub

### Dataset

Titanic Dataset

### Steps Performed

#### 1. Data Loading

* Downloaded and loaded the Titanic dataset.
* Imported the dataset into a Pandas DataFrame.

#### 2. Data Preprocessing

* Checked for missing values.
* Replaced missing values in the Age column using the median.
* Selected relevant features for regression.

#### 3. Feature Selection

Input Features:

* Age
* Pclass

Target Variable:

* Fare

#### 4. Train-Test Split

* Split the dataset into training and testing sets using an 80:20 ratio.

#### 5. Model Training

* Created a Linear Regression model using Scikit-learn.
* Trained the model on the training dataset.

#### 6. Prediction

* Generated fare predictions on the test dataset.

#### 7. Model Evaluation

Evaluated model performance using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

#### 8. Visualization

Created the following plots:

* Actual vs Predicted Fare
* Residual Plot

#### 9. Coefficient Interpretation

* Examined model intercept and feature coefficients.
* Analyzed the impact of Age and Passenger Class on Fare prediction.

### Results

The Linear Regression model was successfully trained and evaluated. The evaluation metrics provided insight into prediction accuracy, while the visualizations helped assess model performance and residual distribution.

### Key Concepts Learned

* Linear Regression
* Feature Selection
* Train-Test Splitting
* Model Training
* Prediction
* MAE, MSE, and R² Metrics
* Model Interpretation
* Residual Analysis

### Outcome

Successfully implemented a Linear Regression model and evaluated its performance using standard regression metrics and visualizations.

### Author

Submitted as part of the AI & ML Internship Program.
