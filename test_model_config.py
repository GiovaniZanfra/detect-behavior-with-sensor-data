#!/usr/bin/env python3
"""
Test script to verify model configuration works for different model types
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import detect_behavior_with_sensor_data.config as config
from detect_behavior_with_sensor_data.modeling.train import create_model, fit_model, get_best_iteration

def test_model_creation():
    """Test that different model types can be created successfully"""
    print("Testing model creation for different model types...")
    
    # Test each supported model type
    model_types = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]
    
    for model_type in model_types:
        try:
            print(f"\nTesting {model_type}...")
            
            # Temporarily change the model type
            original_model_type = config.MODEL_TYPE
            config.MODEL_TYPE = model_type
            
            # Create model
            model = create_model(model_type, num_classes=5)
            print(f"✓ Successfully created {model_type} model")
            
            # Test basic functionality
            import numpy as np
            X = np.random.rand(10, 5)
            y = np.random.randint(0, 5, 10)
            
            # Test fitting
            model = fit_model(model, X, y, model_type=model_type)
            print(f"✓ Successfully fitted {model_type} model")
            
            # Test prediction
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            print(f"✓ Successfully predicted with {model_type} model")
            
            # Test best iteration (for gradient boosting models)
            best_it = get_best_iteration(model, model_type)
            if model_type in ["xgboost", "lightgbm"]:
                print(f"  Best iteration: {best_it}")
            
            # Restore original model type
            config.MODEL_TYPE = original_model_type
            
        except Exception as e:
            print(f"✗ Error with {model_type}: {e}")
            config.MODEL_TYPE = original_model_type

def test_config_parameters():
    """Test that configuration parameters are properly set"""
    print("\nTesting configuration parameters...")
    
    print(f"Current MODEL_TYPE: {config.MODEL_TYPE}")
    print(f"Available model types: {list(config.MODEL_PARAMS.keys())}")
    
    for model_type, params in config.MODEL_PARAMS.items():
        print(f"\n{model_type} parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    print("=== Model Configuration Test ===\n")
    
    test_config_parameters()
    test_model_creation()
    
    print("\n=== Test Complete ===")
    print("\nTo change the model type, edit config.py and set:")
    print("MODEL_TYPE = 'your_choice'  # Options: xgboost, lightgbm, random_forest, logistic_regression")
