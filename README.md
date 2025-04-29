# Car Insurance Lab

This repository contains a number of labs to lilustrate the use of Google Cloud Data and AI products using a retail/e-commerce dataset

## Setup

Once logged into a Google Cloud Project, open Cloud Shell and execute the following commands:

```bash
gcloud services enable bigqueryconnection.googleapis.com
gcloud services enable notebooks.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable iam.googleapis.com

sleep 60


PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUM=$(gcloud projects list --filter="$PROJECT_ID" --format="value(PROJECT_NUMBER)")


gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com"\
      --role='roles/aiplatform.serviceAgent'

gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com"\
      --role='roles/artifactregistry.admin'

gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
        --role='roles/bigquery.connectionAdmin'

gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com"\
      --role='roles/storage.admin'

gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:${PROJECT_NUM}@cloudbuild.gserviceaccount.com"\
      --role='roles/aiplatform.admin'

gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:$PROJECT_NUM-compute@developer.gserviceaccount.com"\
      --role='roles/resourcemanager.projectIamAdmin'

```

Next, navigate to Vertex AI > Workbench and create a new notebook of `INSTANCES` type.

Leave all the configuration as default.
Once the notebook is created, click on `OPEN JUPYTERLAB` to open the web interface.
Click on Git > Clone a Repository and enter the following URL `https://github.com/akanksha86/retail-demo`.

Now, edit the `retail-demo/src/datagen/launch_datagen.sh` script and ammend the values maked with `TO_DO_DEVELOPER`
Now, open a Terminal on the notebook and run the shell script, navigate to the datagen folder and:

```bash
source lanch_datagen.sh
```

This concludes the lab setup.

## Labs

Now you can follow the lab instructions on the following notebooks.

You can run all of them with default Python3 kernel

- Lab #1: BigQuery Data Preparation - use YAML files for upload to Data Preparation
  - [Code](Exchange Rates data preparation.dp.yaml)
  - [Code](Orders data preparation.dp.yaml)
- Lab #2: Data Insights - Generate Insights for Customers table and show table and column level descriptions
- Lab #3: Data Canvas as notebooks - Agentic Data Analysis
  - [Code](src/notebooks/03_revenue_forecast.ipynb)
  
- Lab #3: TimesFM forecasting notebook
  - [Code](src/notebooks/03_revenue_forecast.ipynb)