## Comparative Analysis of Open-Source Large Language Models for Sentiment Analysis and Prompt Engineering
This project involves using various transformer models for sentiment analysis and prompt engineering, focusing on comparing the performance and insights each model provides. The project is divided into two main parts: Sentiment Analysis and Prompt Engineering.

Table of Contents
Overview
Prerequisites
Dataset
Setup
Running the Notebook on Kaggle
Code Structure
Models Used
Results and Evaluation
Troubleshooting
References
Overview
This project aims to perform the following:

Sentiment Analysis:

Implemented using models such as DistilBERT, GPT-2, XLNet, and Electra.
Conducted by freezing the layers of the large language models (LLMs), introducing a classification layer, and using the softmax function to predict positive or negative sentiments.
Prompt Engineering:

Implemented using the LLaMA and Mistral models.
Focuses on analyzing text to extract themes and sentiment drivers for positive and negative reviews.
Prerequisites
Before running the code, ensure you have the following:

A Kaggle account
Basic understanding of Python and machine learning concepts
Familiarity with Jupyter Notebooks and Kaggle’s interface
Dataset
The dataset for this project is used for both sentiment analysis and prompt engineering. It should be placed in a directory named amazon with the file name test.csv. Additionally, the datasets for positive and negative reviews extracted from the predicted sentiments are stored in positive_reviews.csv and negative_reviews.csv, respectively.

Uploading the Dataset on Kaggle
Prepare the Dataset: Ensure your dataset is named test.csv.
Upload the Dataset: Go to the "Datasets" section on Kaggle and upload your test.csv file. Set the dataset title as amazon.
Add the Dataset to the Notebook: In your Kaggle notebook, click on the "Add Data" button and select the amazon dataset.
Setup
To run this project, follow the steps below to set up the environment:

Clone the Repository:

bash
git clone https://github.com/Freddie-2/Capstone-Project.git
cd your-repo-name
Install Required Libraries:

Make sure you have Python 3.6 or above installed. Install the required libraries using pip:

bash
pip install pandas lamini transformers torch sklearn matplotlib seaborn numpy
Set up API Key for Lamini (if applicable):

Obtain your API key from the Lamini website and set it in the script:

lamini.api_key = "your_lamini_api_key"
Place the CSV Files:

Ensure that positive_reviews.csv and negative_reviews.csv are placed in the same directory as your script.

Running the Notebook on Kaggle
To run the notebook on Kaggle, follow these steps:

Open Kaggle and Sign In: Go to Kaggle and log in to your account.
Create a New Notebook: Navigate to the Notebooks section and create a new notebook.
Upload the Code: Upload the Electra_model.ipynb file to your notebook.
Add the Dataset: Ensure the dataset named amazon is added to your notebook’s environment.
Install Required Libraries: If any libraries are missing in the Kaggle environment, install them using the following command at the start of your notebook:
python
Copy code
!pip install transformers torch sklearn matplotlib seaborn numpy pandas
Modify the Data Loading Section: Update the data loading section in your code as follows:
python
Copy code
import pandas as pd

# Load the dataset
df = pd.read_csv("/kaggle/input/amazon/test.csv")
Run the Notebook: Execute the cells in the notebook sequentially. Make sure all cells run without errors.
Code Structure
Here is an overview of the code structure in the notebook:

Importing Libraries: Import all necessary libraries for data processing, model creation, and evaluation.
Data Loading and Preprocessing: Load the dataset and perform necessary preprocessing steps such as tokenization and encoding.
Model Definition and Training: Define each of the four models (DistilBERT, GPT-2, XLNet, and Electra) and train them on the preprocessed data.
Model Evaluation: Evaluate each model using various metrics such as accuracy, confusion matrix, and classification report.
Comparison of Results: Compare the performance of the four models based on the evaluation metrics.
Prompt Engineering Code: Includes the setup, processing of reviews, and generation of insights using the Mistral model from Lamini.
Models Used
The notebook uses the following models:

DistilBERT:

Model: distilbert-base-uncased
Library: Hugging Face Transformers
GPT-2:

Model: gpt2
Library: Hugging Face Transformers
XLNet:

Model: xlnet-base-cased
Library: Hugging Face Transformers
Electra:

Model: google/electra-small-discriminator
Library: Hugging Face Transformers
LLaMA and Mistral for Prompt Engineering:

Model: mistralai/Mistral-7B-Instruct-v0.1 and Meta-Llama-3-8B-Instruct
Library: Lamini
Each model is fine-tuned for sentiment analysis using the dataset and evaluated based on the performance metrics. The Mistral model is used for prompt engineering to extract themes and sentiment drivers.

Results and Evaluation
After running the notebook, you will obtain the following results for each model:

Classification Report: Shows precision, recall, f1-score, and support for each class.
Confusion Matrix: Visual representation of the model’s performance.
Accuracy Score: Overall accuracy of each model.
The script will also output the themes and sentiment drivers for each review, categorized as either positive or negative based on the sentiment analysis results.

Troubleshooting
If you encounter any issues while running the notebook, consider the following tips:

Library Errors: Ensure all libraries are correctly installed. You can use !pip install to install missing libraries.
Dataset Issues: Verify that the dataset is properly loaded and accessible in the notebook environment.
Model Errors: Check the model definition and training process for any discrepancies or errors.

References
DistilBERT: A distilled version of BERT
GPT-2: Language Models are Unsupervised Multitask Learners
XLNet: Generalized Autoregressive Pretraining for Language Understanding
Electra: Pre-training Text Encoders as Discriminators Rather Than Generators
Hugging Face Transformers Documentation
Kaggle Documentation
Lamini Website
