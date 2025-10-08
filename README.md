# Human Activity Recognition (HAR) using Smartphone Data

## Project Overview

This project implements a Human Activity Recognition (HAR) system using sensor data collected from smartphones. The goal is to classify various human activities (such as walking, sitting, standing, etc.) based on readings from accelerometers and gyroscopes. This mini-project was inspired by a Stanford CS229 project and utilizes a publicly available dataset.

### Resources
*   **Original Project Report:** [https://cs229.stanford.edu/proj2015/100_report.pdf](https://cs229.stanford.edu/proj2015/100_report.pdf)
*   **Original Project Poster:** [https://cs229.stanford.edu/proj2015/100_poster.pdf](https://cs229.stanford.edu/proj2015/100_poster.pdf)
*   **Dataset:** [Human Activity Recognition Using Smartphones Data Set (UCI ML Repository)](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

## Dataset Description

The dataset consists of 3-axial linear acceleration and 3-axial angular velocity (gyroscope) captured from Samsung Galaxy S II smartphones. The raw sensor signals were pre-processed and windowed into fixed-width sliding windows of 2.56 seconds with 50% overlap (128 readings per window at 50Hz). From these time-domain signals, 561 features were extracted for each window.

The activities considered are:
*   WALKING
*   WALKING_UPSTAIRS
*   WALKING_DOWNSTAIRS
*   SITTING
*   STANDING
*   LAYING

The dataset is divided into training (70%) and test (30%) sets.

## Project Steps and Methodology

The project followed a standard Machine Learning pipeline:

### 1. Data Loading and Initial Preprocessing
*   Loaded `X_train.txt`, `y_train.txt`, `X_test.txt`, `y_test.txt`, `features.txt`, and `activity_labels.txt`.
*   Assigned descriptive feature names to `X_train` and `X_test` columns.
*   Mapped numerical activity labels (1-6) to their corresponding descriptive names (e.g., 'WALKING').
*   Checked for missing values (none found, confirming data cleanliness).

### 2. Exploratory Data Analysis (EDA) and Scaling
*   Visualized the distribution of activities in the training set, noting a relatively balanced dataset.
*   Applied `StandardScaler` to standardize the features (`X_train` and `X_test`) to have zero mean and unit variance. The scaler was fitted only on `X_train` and then used to transform both `X_train` and `X_test` to prevent data leakage.

### 3. Feature Selection
*   **Objective:** To investigate if a reduced set of features could maintain or improve model performance, and to potentially reduce model complexity.
*   **Method:** A `RandomForestClassifier` was trained on the full scaled training data to calculate feature importances. The top 50 most important features were then selected.
*   **Key Finding:** Features related to `tGravityAcc` (gravity acceleration) and `angle(X,gravityMean)` (angles relative to gravity vector) were found to be highly important, indicating their strong discriminative power for static activities.
*   **Impact:** When supervised models were trained on *only* these top 50 features, a significant **drop in accuracy** (from ~96% to ~89% for Logistic Regression and SVM) was observed. This suggested that while the top features are crucial, the remaining features, though individually less important, collectively contribute valuable information for optimal classification.
    *   *(Note: For the final hyperparameter tuning, models will be re-evaluated with the top 50 features as per project requirement.)*

### 4. Unsupervised Learning: K-Means Clustering
*   **Objective:** To explore the inherent structure of the data and see how well natural groupings (clusters) align with the known activity labels, without prior knowledge of these labels.
*   **Method:** K-Means clustering was applied to the scaled training data (using the top 50 features) with `k=6` (matching the number of true activities).
*   **Evaluation:** A cross-tabulation and heatmap were generated to visualize the overlap between K-Means clusters and true activity labels. Metrics like Homogeneity, Completeness, V-measure, Adjusted Rand Index, and Adjusted Mutual Information were calculated.
*   **Key Insight:** K-Means effectively differentiated between **static activities (LAYING, SITTING, STANDING)** and **dynamic activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS)**. However, it struggled to perfectly distinguish very similar activities within these groups (e.g., SITTING vs. STANDING) based purely on unsupervised patterns.

### 5. Supervised Model Training and Evaluation (on Top 50 Features)
*   **Models Implemented:**
    *   **Logistic Regression**
    *   **Support Vector Machine (SVC)**
*   **Initial Performance (on top 50 features, before tuning):**
    *   Logistic Regression Accuracy: ~0.8884
    *   SVC Accuracy: ~0.8887
*   **Detailed Results for these models will be provided after hyperparameter tuning.**

### 6. Hyperparameter Tuning (using Top 50 Features)
*   **Objective:** To optimize the performance of Logistic Regression and SVC by systematically searching for the best combination of their hyperparameters.
*   **Method:** `GridSearchCV` with 5-fold cross-validation will be used for both models, searching across a predefined parameter grid. Tuning will be performed on the training data using the **top 50 features**.
*   **Parameters Explored:**
    *   **Logistic Regression:** `C` (regularization strength), `solver`.
    *   **SVC:** `C` (regularization parameter), `gamma` (kernel coefficient), `kernel` ('rbf').

## Results (To be updated after tuning)

*(Once you complete the hyperparameter tuning, you will update this section with the final best parameters and performance metrics.)*

### Logistic Regression (Tuned, Top 50 Features)
*   **Best Parameters:** `{...}`
*   **Best Cross-Validation Accuracy:** `...`
*   **Test Set Accuracy:** `...`
*   **Classification Report:**
    ```
    (Paste classification report here)
    ```

### Support Vector Machine (SVC) (Tuned, Top 50 Features)
*   **Best Parameters:** `{...}`
*   **Best Cross-Validation Accuracy:** `...`
*   **Test Set Accuracy:** `...`
*   **Classification Report:**
    ```
    (Paste classification report here)
    ```

## Conclusion

*(This section will summarize your final findings. Which model performed best after tuning? What are the key takeaways regarding feature importance and the capabilities of supervised vs. unsupervised learning for this problem?)*

## How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Download Dataset:** Download `UCI HAR Dataset.zip` from [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and extract its contents into the root of this project directory, so you have a `UCI HAR Dataset/` folder.
3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
    ```
4.  **Run the script:**
    ```bash
    python your_main_script_name.py # or open and run your Jupyter Notebook
    ```

## Future Work (Optional)

*   Experiment with different numbers of features for selection (e.g., top 100, 200).
*   Explore other feature selection methods (e.g., Recursive Feature Elimination).
*   Tune hyperparameters for other models like Random Forest or XGBoost.
*   Implement a simple Neural Network to compare performance.
*   Analyze misclassifications in more detail using a confusion matrix for the best model.
