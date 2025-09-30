# SinkOrSurvive
📊 Dataset
Dataset: Kaggle Titanic: Machine Learning from Disaster
Columns used:
Pclass → ticket class (1–3)
Sex → male / female
Age → filled with median
SibSp → siblings/spouses aboard
Parch → parents/children aboard
Fare → filled with median
Embarked → C, Q, or S (most frequent fill for missing)
HasCabin → binary (was cabin info recorded?)
Engineered Features:
Family_Size = SibSp + Parch + 1
IsAlone = 1 if Family_Size == 1 else 0
🤖 Model Training (train.py)
Model: GradientBoostingClassifier
Hyperparameter tuning with GridSearchCV
Scoring metric: ROC AUC
Feature importance plotted
Example Outputs:
Best params found by GridSearchCV
CV AUC and Test AUC
Confusion matrix heatmap
Feature importance bar chart
The best pipeline is saved as:
titanic_pipeline.joblib
🛠️ FastAPI Backend (app.py)
Run API:
uvicorn app:app --reload --port 8000
Endpoints:
GET /health → simple health check
POST /predict → single passenger prediction
POST /predict-batch → batch predictions (list of passengers)
Example request:
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S",
  "HasCabin": 0
}
Example response:
{
  "prediction": 0,
  "survival_probability": 0.12
}
Docs auto-generated:
Swagger UI: http://127.0.0.1:8000/docs
ReDoc: http://127.0.0.1:8000/redoc
🎛️ Streamlit Frontend (streamlit_frontend.py)
Interactive UI for predictions.
Run:
streamlit run streamlit_frontend.py
Opens at: http://localhost:8501
Features:
Sidebar form for passenger details (class, sex, age, family, fare, etc.)
Calls FastAPI /predict endpoint
Displays prediction + survival probability with ✅ or ❌
📦 Requirements
  fastapi
  uvicorn
  pydantic
  pandas
  scikit-learn
  joblib
  streamlit
  seaborn
  matplotlib
  requests
Install:
  pip install -r requirements.txt
  📈 Visualizations in Training
  Survival by Sex (barplot)
  Survival by Class (barplot)
  Fare vs Survival (violinplot)
  Age vs Survival (stacked histogram)
  Confusion Matrix Heatmap
  Feature Importance Bar Chart
