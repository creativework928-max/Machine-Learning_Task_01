Spotify Personalized Music Recommendation System
1. Project Overview

This project implements an end-to-end machine learning pipeline for a Spotify-style personalized music recommendation system.

The system predicts the likelihood that a user will repeat-listen to a song within a defined period and uses these predictions to rank candidate songs and generate personalized Top-10 recommendations.

The project covers:

Synthetic dataset generation
Data cleaning and validation
Exploratory Data Analysis (EDA)
Feature engineering
Machine learning model training
Time-based model evaluation
Target leakage detection and correction
Recommendation generation
Recommendation visualization

Important: The dataset used in this project is synthetic. Spotify does not publicly provide individual users' private listening histories. The synthetic data was generated to reproduce realistic patterns in music listening behavior and Spotify-style audio features.

2. Project Objectives

The main objectives are:

Build a realistic music listening-history dataset.
Clean and validate the generated data.
Explore listening behavior and audio-feature distributions.
Engineer user, song, and user-song interaction features.
Predict repeat listening behavior.
Compare multiple machine learning models.
Evaluate models using a time-based train/test split.
Detect and eliminate target leakage.
Generate personalized Top-10 recommendations.
3. Dataset

The synthetic dataset contains:

2,000 users
1,500 songs
150,000 listening events
User demographic/profile attributes
Song metadata
Audio-style features
Listening context
Skip behavior
Listening duration ratio
Timestamps

After cleaning and interaction construction:

118,921 user-song interaction records
33 engineered model features
Repeat-listening positive rate: approximately 2.04%
4. Project Pipeline

The project is divided into seven independent scripts.

01 Data Generation
        ↓
02 Data Cleaning
        ↓
03 Exploratory Data Analysis
        ↓
04 Feature Engineering
        ↓
05 Model Training
        ↓
06 Model Evaluation
        ↓
07 Recommendation Engine

Script 01 — Data Generation
scripts/01_data_generation.py


Generates:

Song catalog
User profiles
Listening-history events

Outputs:

data/songs.csv
data/users.csv
data/listening_history.csv

Script 02 — Data Cleaning
scripts/02_data_cleaning.py


Performs:

Duplicate detection
Missing-value checks
Referential-integrity validation
Cleaning of song/user/listening data
Construction of the repeat-listening target
User-song interaction aggregation

Outputs:

data/songs_clean.csv
data/users_clean.csv
data/listening_history_clean.csv
data/interactions.csv

Script 03 — Exploratory Data Analysis
scripts/03_eda_visualization.py


Generates eight visualizations covering:

Target-class distribution
Genre play counts
Repeat rate by genre
Audio-feature distributions
Popularity versus repeat rate
Activity patterns
User-tier behavior
Feature correlations

Outputs are stored in:

outputs/figures/

Script 04 — Feature Engineering
scripts/04_feature_engineering.py


Creates user-level, song-level, and interaction-level predictive features.

Final feature matrix:

118,921 rows
33 model features


Output:

data/model_features.csv

Script 05 — Model Training
scripts/05_model_training.py


Models compared include:

Logistic Regression
Random Forest
Balanced Random Forest
Gradient Boosting

A chronological/time-based split is used to reduce temporal leakage.

Script 06 — Model Evaluation
scripts/06_model_evaluation.py


Produces:

ROC curves
Precision-Recall curves
Confusion matrix
Threshold tuning
Model comparison
Feature importance
Script 07 — Recommendation Engine
scripts/07_recommendation_engine.py


The recommendation pipeline:

Generates candidate songs.
Calculates model scores.
Ranks candidates.
Applies recommendation filtering/diversity logic.
Produces Top-10 recommendations for sample users.

Final output:

outputs/reports/sample_recommendations.csv

5. Model Results

The final evaluation used a time-based test set.

Test set size:

23,785 interactions


Test positive rate:

1.207%

Model comparison
Model	ROC-AUC	PR-AUC	F1
Logistic Regression	0.6869	0.0340	0.0538
Random Forest	0.6726	0.0384	0.0000
Random Forest Balanced	0.6725	0.0374	0.0725
Gradient Boosting	0.6668	0.0275	0.0121

The selected model was Logistic Regression, based on the project's primary model-selection criterion of ROC-AUC.

At the default threshold of 0.5:

ROC-AUC: 0.6869
PR-AUC: 0.0340
Precision: 0.0288
Recall: 0.4042
F1: 0.0538

The evaluation stage selected a recommendation decision threshold of approximately:

0.7711

6. Target Leakage Detection

An initial model produced an ROC-AUC of approximately 0.99.

This was identified as suspicious.

Investigation showed that some aggregate play-count/repeat-related features had been calculated using information from the complete listening history, including information that could occur after the prediction point.

This introduced target leakage.

The feature construction was corrected so that future information was not used to construct predictive features.

After correcting the leakage, the model achieved:

ROC-AUC ≈ 0.69


This result is considered substantially more realistic.

The leakage correction is an important part of the project because it demonstrates proper machine learning validation rather than simply optimizing for a high evaluation score.

7. Recommendation Output

The recommendation engine produces personalized Top-10 recommendations.

Example output fields:

user_id
song_id
track_name
artist_id
genre
popularity
predicted_repeat_prob


Example:

user_id: 1692
track: Track_787
genre: Classical
predicted_repeat_prob: 0.6393


The complete sample output is available at:

outputs/reports/sample_recommendations.csv

8. Installation
Requirements

Recommended environment:

Python 3.14.x
Windows/macOS/Linux
pip
Virtual environment
Create virtual environment

Windows PowerShell:

py -m venv .venv


Activate:

.\.venv\Scripts\Activate.ps1


Install dependencies:

python -m pip install -r requirements.txt

9. Running the Project

Run the scripts in the following order:

python scripts\01_data_generation.py
python scripts\02_data_cleaning.py
python scripts\03_eda_visualization.py
python scripts\04_feature_engineering.py
python scripts\05_model_training.py
python scripts\06_model_evaluation.py
python scripts\07_recommendation_engine.py


Do not skip stages because later scripts depend on outputs generated by earlier stages.

10. Project Outputs
Data
data/


Contains generated, cleaned, and model-ready datasets.

Visualizations
outputs/figures/


Contains the 15 project visualizations.

Models
outputs/models/


Contains trained machine learning pipelines and evaluation summaries.

Recommendations
outputs/reports/sample_recommendations.csv


Contains Top-10 recommendations for sample users.

Report
docs/Spotify_Recommendation_System_Report.docx


Contains the detailed internship project report.

11. Limitations

This project is a machine learning demonstration and not an implementation of Spotify's proprietary recommendation infrastructure.

Important limitations include:

The listening-history dataset is synthetic.
Spotify's proprietary user and recommendation data is not used.
Track metadata/audio features are simulated rather than retrieved from Spotify's private systems.
The model predicts repeat-listening behavior rather than directly optimizing user satisfaction.
Offline evaluation does not fully represent real-world recommendation quality.
The recommendation engine is intended as an academic/portfolio demonstration.
12. Future Improvements

Potential improvements include:

Collaborative filtering
Matrix factorization
Neural collaborative filtering
Sequence-based recommendation models
Transformer-based session modeling
Real-time recommendation updates
More sophisticated diversity optimization
Cold-start handling
Online A/B testing
Ranking metrics such as NDCG@K and MAP@K
Integration with a legitimate external music metadata API where permitted
13. Author

Internship Machine Learning Project

Project: Spotify Personalized Music Recommendation System

Technologies:

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
SciPy
Joblib
python-docx
14. License / Usage

This project is intended for educational, internship, and portfolio purposes.

The dataset is synthetic and does not contain private Spotify user listening histories.