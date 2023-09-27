from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(path):
    df = pd.read_csv(path)
    df.drop_duplicates(subset=['location'], inplace=True)

    attraction_types = df['attraction_type'].str.get_dummies(' • ')
    df = pd.concat([df, attraction_types], axis=1)
    df['Comment'] = df[['Comment_1', 'Comment_2', 'Comment_3']].apply(lambda x: ' '.join(x.dropna()), axis=1)
    df.drop(['Comment_1', 'Comment_2', 'Comment_3', 'attraction_type'], axis=1, inplace=True)
    df['num_reviews'] = df['num_reviews'].str.replace(',', '').astype(int)

    return df

def create_df_cbf(df):
    df = df.drop(['location', 'Comment', 'num_reviews'], axis=1)

    return df
def calculate_cosine_similarity(df, attraction_index):
    cosine_sim_matrix = cosine_similarity(df)
    similarity_scores = cosine_sim_matrix[attraction_index]
    similar_attractions_indices = np.argsort(similarity_scores)[::-1]

    return similarity_scores, similar_attractions_indices


def apply_numreview_weighting(df, similarity_scores):
    normalized_reviews = df['num_reviews'] / df['num_reviews'].max()  # Normalize reviews
    weighted_scores = similarity_scores + normalized_reviews
    similar_attractions_indices = np.argsort(weighted_scores)[::-1]

    return similar_attractions_indices
def index(request):

    # Load and preprocess data
    data = preprocess_data(
        r"C:\Users\Stiliyan\Desktop\mta_project-Dev-b39783a234834d06855fc2211d2fca997ef2fa5c\Recommendation_Engine"
        r"\tripadvisor_poi.csv")
    print(data.head(1))

    # Perform feature engineering
    feature_matrix = create_df_cbf(data)
    print(feature_matrix.head(1))

    # Perform clustering
    # cluster_labels = clustering.perform_clustering(feature_matrix)

    # Generate recommendations
    similarity_scores, similar_attractions_indices = calculate_cosine_similarity(feature_matrix,
                                                                                                attraction_index=1)
    recommendations = apply_numreview_weighting(data, similarity_scores)

    # Print recommendations
    top_n = 5
    recommended_indices = recommendations[1:top_n + 1]
    val = []
    html = "<html><body>"
    html += "Recommended Attractions:"
    i = 0
    for index in recommended_indices:
        i+=1
        html += "<p>"
        html += str(i) + ": "
        html += data.iloc[index]['location']
        html += "<br> "
        html += "</p>"
    return HttpResponse(html)