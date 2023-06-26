from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(request.form)
        # Get the values from the form
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])
        key = int(request.form['key'])
        loudness = float(request.form['loudness'])
        mode = int(request.form['mode'])
        speechiness = float(request.form['speechiness'])
        acousticness = float(request.form['acousticness'])
        instrumentalness = float(request.form['instrumentalness'])
        liveness = float(request.form['liveness'])
        valence = float(request.form['valence'])
        tempo = float(request.form['tempo'])
        duration_ms = int(request.form['duration_ms'])
        
        # Load the saved model, scaler, and encoder
        with open('genre_model.pkl', 'rb') as file:
            model = pickle.load(file)

        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        with open('encoder.pkl', 'rb') as file:
            le = pickle.load(file)

        # Create a function to predict the genre of a song
        def predict_genre(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms):   
            x = [[danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, 0]]  # Add 0 for the missing feature (mode)
            x = scaler.transform(x)
            genre = model.predict(x)
            genre = le.inverse_transform(genre)
            return genre[0]
        
        
        # Predict the genre of the song
        genre = predict_genre(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms)
        print(genre)
        return render_template('index.html', genre=genre)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)





