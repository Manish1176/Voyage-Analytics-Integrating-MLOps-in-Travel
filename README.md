# Hotel Recommendation System 

[Visit Streamlit](https://hotelrecommendation-ltff985abrt4wsuoqog4l2.streamlit.app/)

This project implements a hotel recommendation system using collaborative filtering. The system suggests hotels based on user similarities and their booking patterns.

## How it Works
The recommendation system uses collaborative filtering to suggest hotels:
1. Creates a user-hotel matrix based on booking history
2. Calculates user similarities using cosine similarity
3. Finds similar users for a given user
4. Recommends hotels that similar users have booked

## Dataset
The dataset used in this project encompasses multiple aspects:

- **Gender Classification**: Features that contribute to predicting the gender of a user, which may include demographics and user preferences.
- **Hotel Recommendation**: Data related to user preferences, past bookings, and ratings, utilized for suggesting hotels.
- **Flight Price Prediction**: A model that predicts the price of flights bookings based on various factors, including location, and demand.


## Models

### 1. Gender Classification Model
- **Purpose**: Predicts user gender based on demographics and preferences
- **Features**: Age, Company
- **Model Type**: XGBClassifier
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### 2. Hotel Recommendation Model
- **Purpose**: Recommend hotels to users based on past preferences and behavioral data.
- **Model Type**: Recommendation System
- **Performance Metrics**: Precision, Recall

### 3. Flight Price Prediction Model
- **Purpose**: Predicts flight prices based on route and timing
- **Features**: Origin, Destination, Flight Type, Agency, Time, Distance, Date
- **Model Type**: Random Forest Regressor
- **Metrics**: MSE, MAE, R² Score

## Setup and Running
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Docker Build Command
   docker build -t flight-pridiction .
   docker build -t gender-classification-app .
   
5. Run the container
   docker run -p 5000:5000 flight-pridiction
   docker run -p 5000:5000 gender-classification-app
   
## Usage
1. Enter your user code in the web interface
2. View the top 5 recommended hotels based on your profile

## Data Requirements
- users.xlsx: Contains user information
- hotels.xlsx: Contains hotel booking information

[Visit Streamlit](https://hotelrecommendation-ltff985abrt4wsuoqog4l2.streamlit.app/)
