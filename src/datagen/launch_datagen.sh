#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install required packages
pip install -r requirements.txt

# Set environment variables for GCP
export PROJECT_ID="to-do-developer"
export DATASET_ID="to-do-developer"
export GCP_LOCATION="to-do-developer"

# Run the data generation script
python datagen.py

# Deactivate virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
