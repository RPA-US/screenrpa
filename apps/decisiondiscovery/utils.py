from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(data):
  columns_to_drop = list(filter(lambda x:"TextInput" in x, data.columns))
  data = data.drop(columns=columns_to_drop)
  return data

def create_and_fit_pipeline(X,y, model):
  # define type of columns
  status_columns = list(filter(lambda x:"sta_" in x, X.columns))
  one_hot_columns = list(X.select_dtypes(include=['object']).columns.drop(status_columns))
  numeric_features = X.select_dtypes(include=['number']).columns

  # create each transformer
  status_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
                                ('label_encoder', OrdinalEncoder())
                                ])
  one_hot_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
                                ('one_hot_encoder', OneHotEncoder())
                                ])

  numeric_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='mean')),
                                ])#('scaler',StandardScaler())

  # create preprocessor
  preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('one_hot_categorical', one_hot_transformer, one_hot_columns),
        ('status_categorical', status_transformer, status_columns)
    ]
  )

  # create pipeline
  pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model',model)
  ])

  # fit pipeline
  pipeline.fit(X,y)

  return pipeline
