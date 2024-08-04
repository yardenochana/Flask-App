import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from car_data_prep import prepare_data

# קריאה ל-CSV וטעינת הנתונים ל-DataFrame
df = pd.read_csv("dataset.csv")

# הכנת הנתונים באמצעות הפונקציה שהועברה
df_prepared = prepare_data(df)

X = df_prepared.drop('Price', axis=1)
y = df_prepared['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_types = ['int', 'int16', 'int32', 'int64', 'float', 'float16', 'float32', 'float64']
numerical_features = X.select_dtypes(include=numerical_types).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# יצירת המודל ElasticNet עם הפרמטרים הכי טובים
final_elastic_net = ElasticNet(alpha=0.001, l1_ratio=0.5)

# עדכון הפייפליין עם המודל הכי טוב
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_elastic_net)
])

# התאמת הפייפליין הסופי על נתוני האימון
final_pipeline.fit(X_train, y_train)

pickle.dump(final_pipeline, open("trained_model.pkl", "wb"))