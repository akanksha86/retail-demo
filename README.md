# Nordics Retailer - Data Agents

This repository contains a number of labs to lilustrate the use of Google Cloud Data and AI products using a retail/e-commerce dataset mainly showing what Data Agents and Agentic workflows could do for Data Roles

## Setup

Enable all the relevant APIs to work with BigQuery and Vertex AI, either create a new Service Account or use existing Service Account and provide permissions needed to run Data Preparation, Pipeline, Canvas and Insights. 

First step is to create the data to be used to show these capabilities

Edit the `retail-demo/src/datagen/launch_datagen.sh` script and ammend the values maked with `TO_DO_DEVELOPER`
Now, open a Terminal on the notebook and run the shell script, navigate to the datagen folder and:

```bash
source lanch_datagen.sh
```

This would ensure all the data needed to run the other code is in place

## Labs

Now you can follow the lab instructions to show the following capabilities

- Lab #1: BigQuery Data Preparation - use YAML files for upload to Data Preparation
  - [Code](retail-demo/src/other%20code/Exchange%20Rates%20data%20preparation.dp.yaml)
  - [Code](retail-demo/src/other%20code/Orders%20data%20preparation.dp.yaml)
- Lab #2: Data Insights - Generate Insights for Customers table and show table and column level descriptions
- Lab #3: Data Canvas as notebooks - Agentic Data Analysis
  - [Code](retail-demo/src/other%20code/Orders%20notebook.ipynb)
  - [Code](retail-demo/src/other%20code/Sales%20and%20Customers.ipynb)
- Lab #3: TimesFM forecasting notebook
  - [Code](retail-demo/src/other%20code/Sales%20Forecasting%20and%20CA.ipynb)
- Lab #4: AI.Generate and Vector Search
  - [Code](retail-demo/src/other%20code/Product%20Descriptions%20generation.sql)