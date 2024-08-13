import pandas as pd
import lamini
from transformers import pipeline

# Set the API key for lamini
lamini.api_key = "my key"

# Initialize the Lamini model
# llm_lamini = lamini.Lamini("meta-llama/Meta-Llama-3-8B-Instruct")
llm_lamini = lamini.Lamini("mistralai/Mistral-7B-Instruct-v0.1")

# Load the CSV files
positive_reviews = pd.read_csv("positive_reviews.csv")
negative_reviews = pd.read_csv("negative_reviews.csv")

# Define other models you want to compare
models = {
    # "Meta-Llama-3-8B-Instruct (lamini)": llm_lamini,
    "mistralai/Mistral-7B-Instruct-v0.1 (lamini)": llm_lamini,
}

# Define the function to generate insights using the model
def generate_insights_lamini(review, sentiment, model):
    if sentiment == 'positive':
        theme_prompt = f"List the two main topics discussed in this positive review: {review}"
        driver_prompt = f"Explain briefly in one sentence why this review is positive:: {review}"
    else:
        theme_prompt = f"List the two main topics discussed in this negative review: {review}"
        driver_prompt = f"Explain briefly in one sentence why this review is negative:: {review}"
    
    theme_response = model.generate(theme_prompt)
    driver_response = model.generate(driver_prompt)
    
    return theme_response, driver_response

# Function to process reviews and print insights
def process_reviews(reviews, sentiment, model_name, model, num_reviews=None):
    if num_reviews is None:
        num_reviews = len(reviews)
    else:
        num_reviews = min(num_reviews, len(reviews))
    
    print(f"{sentiment.capitalize()} Reviews Insights using {model_name} (Analyzing {num_reviews} reviews):")
    for review in reviews['Review'].head(num_reviews):  # Corrected to 'Review' and limited to num_reviews
        theme, driver = generate_insights_lamini(review, sentiment, model)
        print(f"Review: {review}")
        print(f"Themes: {theme}")
        print(f"Sentiment Drivers: {driver}")
        print()

# Example: Process first 1 positive and 1 negative reviews for each model
num_reviews_to_analyze = 1  # Set the number of reviews to analyze

for model_name, model in models.items():
    process_reviews(positive_reviews, 'positive', model_name, model, num_reviews_to_analyze)
    process_reviews(negative_reviews, 'negative', model_name, model, num_reviews_to_analyze)
