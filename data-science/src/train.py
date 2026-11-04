# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=_______, help="Path to train dataset")  # Specify the type for train_data
    parser.add_argument("--test_data", type=_______, help="Path to test dataset")  # Specify the type for test_data
    parser.add_argument("--model_output", type=_______, help="Path of output model")  # Specify the type for model_output
    parser.add_argument('--n_estimators', type=_______, default=_____,
                        help='The number of trees in the forest')  # Specify the type and default value for n_estimators
    parser.add_argument('--max_depth', type=_______, default=_______,
                        help='The maximum depth of the tree')  # Specify the type and default value for max_depth

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Read train and test data from _______
    train_df = pd.________(Path(args.train_data) / "______.csv")
    test_df = pd.________(Path(args.test_data) / "______.csv")

    # Split the data into ______(X) and ______(y) 
    y_train = train_df['______']  # Specify the target column
    X_train = train_df.drop(columns=['______'])
    y_test = test_df['______']
    X_test = test_df.drop(columns=['______'])

    # Initialize and train a RandomForest Regressor
    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.______, random_state=42)  # Provide the arguments for RandomForestRegressor
    model.________(X_train, y_train)  # Train the model

    # Log model hyperparameters
    mlflow.log_param("model", "_________")  # Provide the model name
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.______)

    # Predict using the RandomForest Regressor on test data
    yhat_test = model._______(X_test)  # Predict the test data

    # Compute and log mean squared error for test data
    mse = mean_squared_error(y_test, yhat_test)
    print('Mean Squared Error of RandomForest Regressor on test set: {:.2f}'.format(mse))
    mlflow.log_metric("MSE", float(mse))  # Log the MSE

    # Save the model
    mlflow.sklearn.________(sk_model=model, path=args.model_output)  # Save the model

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

