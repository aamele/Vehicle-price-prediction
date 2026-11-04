# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=_____, help='Name under which model will be registered')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=_____, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=_____, help="Path to write model info JSON")  # Hint: Specify the type for model_info_output_path (str)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)

    # Load model
    model = mlflow.sklearn.load_model(args._______)  # Load the model from model_path

    # Log model using mlflow
    mlflow.sklearn._______(model, args._______)  # Log the model using with model_name

    # Register logged model using mlflow
    run_id = mlflow.active_run().info.run_id
    model_uri = f'runs:/{run_id}/{args.model_name}'
    mlflow_model = mlflow.__________(model_uri, args._______)  # register the model with model_uri and model_name
    model_version = mlflow_model._______  # Get the version of the registered model

    # Write model info
    print("Writing JSON")
    model_info = {"id": f"{args.model_name}:{model_version}"}
    output_path = os.path.join(args.model_info_output_path, "________.json")  # Specify the name of the JSON file (model_info.json)
    with open(output_path, "w") as of:
        json._____(model_info, of)  # write model_info to the output file

if __name__ == "__main__":
    
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    lines = [
        f"Model name: {args.________}",
        f"Model path: {args.________}",
        f"Model info output path: {args.________}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()