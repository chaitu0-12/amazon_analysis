from flask import Flask, render_template, request
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer

# Define the SentimentModel class here again
class SentimentModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text):
        score = self.analyzer.polarity_scores(text)
        return score['compound']

# Initialize the app
app = Flask(__name__)

# Now load the trained model
with open('Amazon_sentiment_model.sav', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    color = None

    if request.method == 'POST':
        review = request.form['review']
        score = model.predict(review)

        if score >= 0.05:
            sentiment = "Positive"
            color = "green"
        elif score <= -0.05:
            sentiment = "Negative"
            color = "red"
        else:
            sentiment = "Neutral"
            color = "black"

    return render_template('index.html', sentiment=sentiment, color=color)

if __name__ == '__main__':
    app.run(debug=True)
