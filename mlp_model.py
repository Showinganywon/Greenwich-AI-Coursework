from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np



housing=fetch_california_housing()
features, targets=housing.data, housing.target

features_train, features_test, targets_train, targets_test = train_test_split( features, targets, test_size=0.2, random_state=42)

mlp_pipeline=make_pipeline(StandardScaler(), MLPRegressor(random_state=42))
mlp_pipeline.fit(features_train, targets_train)

predictions=mlp_pipeline.predict(features_test)

mae=mean_absolute_error(targets_test, predictions)
rmse=np.sqrt(mean_squared_error(targets_test, predictions))
r2=r2_score(targets_test, predictions)

print("---- MLPRegressor Results ----")
print("MAE:", mae)
print("RMSE:", rmse)
print("R^2:", r2)
