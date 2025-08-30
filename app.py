from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load datasets
Books = pd.read_csv('Books.csv.zip')
Users = pd.read_csv('Users.csv.zip')
Ratings = pd.read_csv('Ratings.csv.zip')

# Preprocess data
ratings_with_name = Ratings.merge(Books, on='ISBN')
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].agg(lambda x: x.astype(float).mean()).reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)

# Get top books with complete info for the homepage
top_books = popular_df.merge(Books.drop_duplicates('Book-Title'), on='Book-Title').head(12)
top_books = top_books[['Book-Title', 'Book-Author', 'Image-URL-M', 'avg_rating']]
top_books = top_books.rename(columns={
    'Book-Title': 'title',
    'Book-Author': 'author',
    'Image-URL-M': 'image_url',
    'avg_rating': 'rating'
})

# Filter active users and famous books
x = Ratings.groupby('User-ID').count()['Book-Rating'] > 200
active_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)

similarity_score = cosine_similarity(pt)

# Function to get recommendations
def recommend(book_name):
    if book_name not in pt.index:
        return []
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    recommendations = [pt.index[i[0]] for i in similar_items]
    return recommendations

# Routes
@app.route('/')
def index():
    return render_template('index.html', books=popular_df['Book-Title'].tolist(), top_books=top_books.to_dict('records'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit-contact', methods=['POST'])
def submit_contact():
    data = request.get_json()
    # In a real app, you would process and store the contact form data here
    return jsonify({'status': 'success', 'message': 'Thank you for your message!'})

@app.route('/recommend', methods=['POST'])
def recommend_books():
    data = request.get_json()
    book_name = data.get('book_name', '')
    recommendations = recommend(book_name)
    results = []
    for rec in recommendations:
        book_info = Books[Books['Book-Title'] == rec].iloc[0]
        average_rating = avg_rating_df[avg_rating_df['Book-Title'] == rec]['avg_rating'].values[0]
        results.append({
            'title': rec,
            'author': book_info['Book-Author'],
            'image_url': book_info['Image-URL-M'],
            'rating': round(average_rating, 2)
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
