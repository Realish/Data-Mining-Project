import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, train_test_split

def load_data():
    # Load the data
    df_train = pd.read_csv("/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/gtzan/Data/features_30_sec.csv")
    df_test_2_1 = pd.read_csv("/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/2:1.csv")
    df_test_10_1 = pd.read_csv("/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/10:1.csv")
    return df_train, df_test_2_1, df_test_10_1

def select_features(df_train, selected_features):
    X_train, X_val, y_train, y_val = train_test_split(df_train[selected_features], df_train['label'], test_size=0.05, random_state=42)
    return X_train, X_val, y_train, y_val

def standardize_features(X_train, X_val, X_test_2_1, X_test_10_1):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_2_1_scaled = scaler.transform(X_test_2_1)
    X_test_10_1_scaled = scaler.transform(X_test_10_1)
    return X_train_scaled, X_val_scaled, X_test_2_1_scaled, X_test_10_1_scaled

def tune_svm_model(X_train_scaled, y_train):
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
    best_hyperparameters = grid_search.best_params_

    return best_svm_model, best_hyperparameters

def evaluate_model(model, X_val_scaled, y_val, X_test_2_1_scaled, y_test_2_1, X_test_10_1_scaled, y_test_10_1):
    # Evaluate the model performance on the validation set
    y_pred_val = model.predict(X_val_scaled)
    recall_val = recall_score(y_val, y_pred_val, average='weighted')
    f1_val = f1_score(y_val, y_pred_val, average='weighted')
    accuracy_val = accuracy_score(y_val, y_pred_val)

    # Make predictions on the test set (2:1 ratio)
    y_pred_2_1 = model.predict(X_test_2_1_scaled)
    recall_2_1 = recall_score(y_test_2_1, y_pred_2_1, average='weighted')
    f1_2_1 = f1_score(y_test_2_1, y_pred_2_1, average='weighted')
    accuracy_2_1 = accuracy_score(y_test_2_1, y_pred_2_1)

    # Make predictions on the test set (10:1 ratio)
    y_pred_10_1 = model.predict(X_test_10_1_scaled)
    recall_10_1 = recall_score(y_test_10_1, y_pred_10_1, average='weighted')
    f1_10_1 = f1_score(y_test_10_1, y_pred_10_1, average='weighted')
    accuracy_10_1 = accuracy_score(y_test_10_1, y_pred_10_1)

    return {
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

def genre_wise_evaluation(model, X_val_scaled, y_val, X_test_2_1_scaled, y_test_2_1, X_test_10_1_scaled, y_test_10_1):
    genre_results = {}

    # Genre-wise evaluation for the validation set
    for genre in pd.unique(y_val):
        genre_indices = y_val == genre
        y_pred_val_genre = model.predict(X_val_scaled[genre_indices])
        genre_accuracy_val = accuracy_score(y_val[genre_indices], y_pred_val_genre)

        genre_results[genre] = {
            'Accuracy (Validation Set)': genre_accuracy_val,
            'Iterations': []
        }

    # Genre-wise evaluation for the test set (2:1 ratio)
    for genre in pd.unique(y_test_2_1):
        genre_indices_2_1 = y_test_2_1 == genre
        y_pred_2_1_genre = model.predict(X_test_2_1_scaled[genre_indices_2_1])
        genre_accuracy_2_1 = accuracy_score(y_test_2_1[genre_indices_2_1], y_pred_2_1_genre)

        genre_results[genre]['Iterations'].append({
            'Accuracy (2:1 ratio)': genre_accuracy_2_1
        })

    # Genre-wise evaluation for the test set (10:1 ratio)
    for genre in pd.unique(y_test_10_1):
        genre_indices_10_1 = y_test_10_1 == genre
        y_pred_10_1_genre = model.predict(X_test_10_1_scaled[genre_indices_10_1])
        genre_accuracy_10_1 = accuracy_score(y_test_10_1[genre_indices_10_1], y_pred_10_1_genre)

        genre_results[genre]['Iterations'].append({
            'Accuracy (10:1 ratio)': genre_accuracy_10_1
        })

    return genre_results

def main():
    df_train, df_test_2_1, df_test_10_1 = load_data()

    # Define the specific features you want to use for testing
    selected_features = [
        'mfcc6_mean', 'mfcc3_mean', 'chroma_stft_var', 'mfcc5_var',
        'mfcc4_var', 'mfcc4_mean', 'mfcc1_mean', 'rms_mean',
        'mfcc19_mean', 'mfcc6_var'
    ]

    X_train, X_val, y_train, y_val = select_features(df_train, selected_features)

    X_test_2_1 = df_test_2_1[selected_features]
    y_test_2_1 = df_test_2_1['genre']

    X_test_10_1 = df_test_10_1[selected_features]
    y_test_10_1 = df_test_10_1['genre']

    X_train_scaled, X_val_scaled, X_test_2_1_scaled, X_test_10_1_scaled = standardize_features(
        X_train, X_val, X_test_2_1, X_test_10_1
    )

    # Number of iterations for averaging
    num_iterations = 10
    results_list = []
    genre_results_list = []

    for iteration in range(num_iterations):
        # Print the selected feature names
        print(f"\nIteration {iteration + 1} - Selected Features:")
        print(X_train.columns)

        best_svm_model, best_hyperparameters = tune_svm_model(X_train_scaled, y_train)

        # Evaluate the model
        evaluation_results = evaluate_model(best_svm_model, X_val_scaled, y_val, X_test_2_1_scaled, y_test_2_1, X_test_10_1_scaled, y_test_10_1)
        results_list.append({'Iteration': iteration + 1, 'Best Hyperparameters': best_hyperparameters, **evaluation_results})

        # Genre-wise evaluation
        genre_results = genre_wise_evaluation(best_svm_model, X_val_scaled, y_val, X_test_2_1_scaled, y_test_2_1, X_test_10_1_scaled, y_test_10_1)
        genre_results_list.append({'Iteration': iteration + 1, 'Genre Results': genre_results})

    # Convert the results to DataFrames
    results_df = pd.DataFrame(results_list)
    genre_results_df = pd.DataFrame(genre_results_list)

    # Save the results to CSV files
    results_csv_path = '/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/results/results_average.csv'
    genre_results_csv_path = '/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/results/genre_results_average.csv'

    results_df.to_csv(results_csv_path, index=False)
    genre_results_df.to_csv(genre_results_csv_path, index=False)

    # Display the average results (excluding 'Best Hyperparameters' column)
    average_results = results_df.drop(['Iteration', 'Best Hyperparameters'], axis=1).mean()
    print("\nAverage Results:")
    print(average_results)
    print(f"Results saved to: {results_csv_path}")
    print(f"Genre-wise results saved to: {genre_results_csv_path}")

if __name__ == "__main__":
    main()
