# Big Data ML Implementations

## Datasets

- Heart Disease (statlog_heart.csv): https://figshare.com/ndownloader/files/34179327
- Airline Customer Satisfaction (airline.csv): https://www.kaggle.com/datasets/sjleshrac/airlines-customer-satisfaction/download?datasetVersionNumber=1
- Credit Card Fraud (fraudTrain.csv): https://www.kaggle.com/datasets/kartik2112/fraud-detection/download?datasetVersionNumber=1
- Particle Classification (SUSY.csv): https://www.kaggle.com/datasets/kartik2112/fraud-detection/download?datasetVersionNumber=1

## How to run scikit-learn script

1. Download the local.py python file
2. Download the above datasets and place them in a directory called resources
3. Install necessary packages (sklearn, pandas, numpy)
4. Run the script to view results

## How to run Spark script (requires AWS Cloud)

1. Upload the above datasets to an S3 bucket
2. Set up AWS EMR with the default configuration and 2 core nodes at minimum
3. Download pyspark, numpy, and setuptools on the EMR cluster
4. Create an EMR notebook, connect it to the cluster, and upload SparkJob.ipynb
5. Set the S3 bucket URLs to the locations of each dataset
6. Run the script to view the results. Change the read{DatasetName}Dataset method to test different datasets
