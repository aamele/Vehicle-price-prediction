# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, help="Path to raw data")  # Specify the type for raw data (str)
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train data (str)
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test data (str)
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")  # Specify the type (float) and default value (0.2) for test-train ratio
    args = parser.parse_args()

    return args

def main(args):  # Write the function name for the main data preparation logic
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # Clean the categorical feature so spacing/capitalisation differences don't create new categories
    df['Segment'] = df['segment'].str.strip().str.lower()

    # Encode categorical feature
    le = LabelEncoder()
    df['Segment'] = le.fit_transform(df['Segment'])  # Write code to encode the categorical feature

    # Split Data into train and test datasets
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)  #  Write code to split the data into train and test datasets

    # Save the train and test data
    os.makedirs(args.train_data, exist_ok=True)  # Create directories for train_data and test_data
    os.makedirs(args.test_data, exist_ok=True)  # Create directories for train_data and test_data
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)  # Specify the name of the train data file
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)  # Specify the name of the test data file

    # log the metrics
    mlflow.log_metric('train size', train_df.shape[0])  # Log the train dataset size
    mlflow.log_metric('test size', test_df.shape[0])  # Log the test dataset size

if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()  # Call the function to parse arguments

    lines = [
        f"Raw data path: {args.raw_data}",  # Print the raw_data path
        f"Train dataset output path: {args.train_data}",  # Print the train_data path
        f"Test dataset path: {args.test_data}",  # Print the test_data path
        f"Test-train ratio: {args.test_train_ratio}",  # Print the test_train_ratio
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
