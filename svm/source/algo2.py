import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, train_test_split

# Load the data
df_train = pd.read_csv("/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/gtzan/Data/features_30_sec.csv")
df_test_2_1 = pd.read_csv("/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/2:1.csv")
df_test_10_1 = pd.read_csv("/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/10:1.csv")

# Define the specific features you want to use for testing
selected_features = [
    'mfcc6_mean', 'mfcc3_mean', 'chroma_stft_var', 'mfcc5_var',
    'mfcc4_var', 'mfcc4_mean', 'mfcc1_mean', 'rms_mean',
    'mfcc19_mean', 'mfcc6_var'
]

# Extract only the selected features and the label for training
X_train, X_val, y_train, y_val = train_test_split(df_train[selected_features], df_train['label'], test_size=0.05, random_state=42)

# Extract only the selected features and the label for testing (2:1 ratio)
X_test_2_1 = df_test_2_1[selected_features]
y_test_2_1 = df_test_2_1['genre']

# Extract only the selected features and the label for testing (10:1 ratio)
X_test_10_1 = df_test_10_1[selected_features]
y_test_10_1 = df_test_10_1['genre']

# Standardize the features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_2_1_scaled = scaler.transform(X_test_2_1)
X_test_10_1_scaled = scaler.transform(X_test_10_1)
genre_accuracies = {}  # Dictionary to store accuracies for each genre



# Number of iterations for averaging
num_iterations = 10
results_list = []

for iteration in range(num_iterations):
    # Print the selected feature names
    print(f"\nIteration {iteration + 1} - Selected Features:")
    print(X_train.columns)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=pd.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(pd.unique(y_train), class_weights))

    # Initialize SVM model
    svm_model = SVC(class_weight=class_weight_dict)

    # Define scoring metrics (recall, F1 score, and accuracy)
    scoring = {'recall': make_scorer(recall_score, average='weighted'),
               'f1_score': make_scorer(f1_score, average='weighted'),
               'accuracy': make_scorer(accuracy_score)}

    # Hyperparameter tuning using grid search
    param_grid = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': ['scale', 'auto', 0.1, 1, 10]}  # Adjust gamma for RBF kernel
    grid_search = GridSearchCV(svm_model, param_grid, scoring=scoring, cv=5, n_jobs=-1, refit='f1_score')
    grid_search.fit(X_train_scaled, y_train)

    # Extract the best model
    best_svm_model = grid_search.best_estimator_

     # Make predictions on the validation set
    y_pred_val = best_svm_model.predict(X_val_scaled)

    # Calculate accuracy for each genre in the validation set
    for genre in df_train['label'].unique():
        genre_indices = y_val == genre
        genre_accuracy = accuracy_score(y_val[genre_indices], y_pred_val[genre_indices])
        
        if genre not in genre_accuracies:
            genre_accuracies[genre] = []
        
        genre_accuracies[genre].append(genre_accuracy)


    # Evaluate the model performance on the validation set
    recall_val = recall_score(y_val, y_pred_val, average='weighted')
    f1_val = f1_score(y_val, y_pred_val, average='weighted')
    accuracy_val = accuracy_score(y_val, y_pred_val)

    # Make predictions on the test set (2:1 ratio)
    y_pred_2_1 = best_svm_model.predict(X_test_2_1_scaled)

    # Evaluate the model performance (2:1 ratio)
    recall_2_1 = recall_score(y_test_2_1, y_pred_2_1, average='weighted')
    f1_2_1 = f1_score(y_test_2_1, y_pred_2_1, average='weighted')
    accuracy_2_1 = accuracy_score(y_test_2_1, y_pred_2_1)

    # Make predictions on the test set (10:1 ratio)
    y_pred_10_1 = best_svm_model.predict(X_test_10_1_scaled)

    # Evaluate the model performance (10:1 ratio)
    recall_10_1 = recall_score(y_test_10_1, y_pred_10_1, average='weighted')
    f1_10_1 = f1_score(y_test_10_1, y_pred_10_1, average='weighted')
    accuracy_10_1 = accuracy_score(y_test_10_1, y_pred_10_1)

    # Extract the best hyperparameters as a dictionary
    best_hyperparameters = grid_search.best_params_

    # Store the results in a dictionary
    results_dict = {
        'Iteration': iteration + 1,
        'Best Hyperparameters': str(best_hyperparameters),
        'Recall (Validation Set)': recall_val,
        'F1 Score (Validation Set)': f1_val,
        'Accuracy (Validation Set)': accuracy_val,
        'Recall (2:1 ratio)': recall_2_1,
        'F1 Score (2:1 ratio)': f1_2_1,
        'Accuracy (2:1 ratio)': accuracy_2_1,
        'Recall (10:1 ratio)': recall_10_1,
        'F1 Score (10:1 ratio)': f1_10_1,
        'Accuracy (10:1 ratio)': accuracy_10_1
    }

    results_list.append(results_dict)

# Calculate average accuracy for each genre
average_genre_accuracies = {genre: sum(accuracies) / num_iterations for genre, accuracies in genre_accuracies.items()}

# Print average accuracy for each genre
print("\nAverage Accuracy for Each Genre:")
for genre, accuracy in average_genre_accuracies.items():
    print(f"{genre}: {accuracy}")

# Convert the results to a DataFrame
results_df = pd.DataFrame(results_list)

# Save the results to a CSV file
results_csv_path = '/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/results/results_average.csv'
results_df.to_csv(results_csv_path, index=False)

# Display the average results (excluding 'Best Hyperparameters' column)
average_results = results_df.drop(['Iteration', 'Best Hyperparameters'], axis=1).mean()
print("\nAverage Results:")
print(average_results)
print(f"Results saved to: {results_csv_path}")
