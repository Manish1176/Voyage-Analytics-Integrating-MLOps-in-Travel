# Hotel Recommendation System

This project implements a hotel recommendation system using collaborative filtering. The system suggests hotels based on user similarities and their booking patterns.

## Project Structure
```
Hotel Recommendation/
├── src/
│   ├── recommender.py    # Core recommendation engine
│   └── app.py           # Streamlit web application
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## How it Works
The recommendation system uses collaborative filtering to suggest hotels:
1. Creates a user-hotel matrix based on booking history
2. Calculates user similarities using cosine similarity
3. Finds similar users for a given user
4. Recommends hotels that similar users have booked

## Setup and Running
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   cd src
   streamlit run app.py
   ```

## Usage
1. Enter your user code in the web interface
2. View the top 5 recommended hotels based on your profile

## Data Requirements
- users.xlsx: Contains user information
- hotels.xlsx: Contains hotel booking information
