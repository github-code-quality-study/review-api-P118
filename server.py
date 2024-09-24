import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize global variables
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
reviews = pd.read_csv('data/reviews.csv').to_dict('records')
allowed_locations = ['San Diego, California', 'Denver, Colorado', 'New York, New York']

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body: str) -> dict:
        """Analyze the sentiment of the review body."""
        return sia.polarity_scores(review_body)

    def filter_reviews(self, location: str, start_date: str, end_date: str) -> list:
        """Filter reviews based on location and date range."""
        filtered_reviews = reviews

        if location:
            filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

        if start_date:
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= datetime.strptime(start_date, '%Y-%m-%d')]

        if end_date:
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= datetime.strptime(end_date, '%Y-%m-%d')]

        return filtered_reviews

    def handle_get_request(self, environ: dict, start_response: Callable) -> bytes:
        """Handle GET requests."""
        query_params = parse_qs(environ.get('QUERY_STRING', ''))
        location = query_params.get('location', [None])[0]
        start_date = query_params.get('start_date', [None])[0]
        end_date = query_params.get('end_date', [None])[0]

        filtered_reviews = self.filter_reviews(location, start_date, end_date)

        for review in filtered_reviews:
            review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

        filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

        response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])
        
        return [response_body]

    def handle_post_request(self, environ: dict, start_response: Callable) -> bytes:
        """Handle POST requests."""
        try:
            request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            request_body = environ['wsgi.input'].read(request_body_size).decode('utf-8')
            review_data = parse_qs(request_body)

            location = review_data.get('Location', [None])[0]
            review_body = review_data.get('ReviewBody', [None])[0]

            if not location or not review_body:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Location and ReviewBody are required"}).encode("utf-8")]

            if location not in allowed_locations:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Invalid location"}).encode("utf-8")]

            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sentiment = self.analyze_sentiment(review_body)

            new_review = {
                'ReviewId': review_id,
                'Location': location,
                'ReviewBody': review_body,
                'Timestamp': timestamp,
                'sentiment': sentiment
            }

            reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")

            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "application/json")])
            return [json.dumps({"error": str(e)}).encode("utf-8")]

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """Handle incoming HTTP requests."""
        if environ["REQUEST_METHOD"] == "GET":
            return self.handle_get_request(environ, start_response)
        elif environ["REQUEST_METHOD"] == "POST":
            return self.handle_post_request(environ, start_response)
        else:
            start_response("405 Method Not Allowed", [("Content-Type", "application/json")])
            return [json.dumps({"error": "Method not allowed"}).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()