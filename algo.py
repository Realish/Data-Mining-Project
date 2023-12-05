import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Load the data
df = pd.read_csv("/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/gtzan/Data/features_30_sec.csv")

# Split the data into features (X) and labels (y)
X = df.drop(['filename', 'label'], axis=1)
y = df['label']

# Number of iterations for averaging
num_iterations = 10
results_list = []

for iteration in range(num_iterations):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=iteration)

    # Standardize the features (optional but often recommended for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=pd.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(pd.unique(y_train), class_weights))

    # Initialize SVM model
    svm_model = SVC(class_weight=class_weight_dict)

    # Define scoring metrics (recall and F1 score)
    scoring = {'recall': make_scorer(recall_score, average='weighted'),
               'f1_score': make_scorer(f1_score, average='weighted')}

    # Hyperparameter tuning using grid search
    param_grid = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': ['scale', 'auto', 0.1, 1, 10]}  # Adjust gamma for RBF kernel
    grid_search = GridSearchCV(svm_model, param_grid, scoring=scoring, cv=5, n_jobs=-1, refit='f1_score')
    grid_search.fit(X_train_scaled, y_train)

    # Extract the best model
    best_svm_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_svm_model.predict(X_test_scaled)

    # Evaluate the model performance
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Extract the best hyperparameters as a dictionary
    best_hyperparameters = grid_search.best_params_

    # Store the results in a dictionary
    results_dict = {'Iteration': iteration + 1,
                    'Best Hyperparameters': str(best_hyperparameters),  # Convert to string to store in DataFrame
                    'Recall': recall,
                    'F1 Score': f1}

    results_list.append(results_dict)

# Convert the results to a DataFrame
results_df = pd.DataFrame(results_list)

# Save the results to a CSV file
results_df.to_csv('/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/results/results.csv', index=False)

# Display the average results (excluding 'Best Hyperparameters' column)
average_results = results_df.drop('Best Hyperparameters', axis=1).mean()
print("\nAverage Results:")
print(average_results)
