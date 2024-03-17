from sklearn.svm import SVC
from datetime import datetime
from Printer import Printer
from DataHelper import DataHelper
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline


printer = Printer(enabled=False)
dataHelper = DataHelper(show_tables=False, printer=printer)

df = pd.read_csv("../data/diabetes_prediction_dataset.csv")

dataHelper.showInitData(df)

df_encoded = dataHelper.format_obj_col(df)

X_balanced, y_balanced = dataHelper.smote_resample(df_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

X_train_clean, y_train_clean = dataHelper.IQR(X_train, y_train)

# MODELS:


# Create an RBF network
current_time = datetime.now()
print(" Starting RBF", current_time.strftime("%Y-%m-%d %H:%M:%S"))
rbf_feature = RBFSampler(gamma=1, random_state=1)  # Consider adjusting gamma
svm_linear_for_rbf = SVC(kernel='linear')

rbf_model = make_pipeline(rbf_feature, svm_linear_for_rbf)
rbf_model.fit(X_train_clean, y_train_clean)

rbf_predictions = rbf_model.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_predictions)
print(f'RBF Network Accuracy: {rbf_accuracy:.2f}')


# SVM MODELS
current_time = datetime.now()
print("Starting SVM linear", current_time.strftime("%Y-%m-%d %H:%M:%S"))
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_clean, y_train_clean)

svm_linear_predictions = svm_linear.predict(X_test)
svm_linear_accuracy = accuracy_score(y_test, svm_linear_predictions)
print(f'SVM Linear Kernel Accuracy: {svm_linear_accuracy:.2f}')

current_time = datetime.now()
print(" Starting SVM nonlinear", current_time.strftime("%Y-%m-%d %H:%M:%S"))
svm_rbf = SVC(kernel='rbf', gamma='auto', C=1)
svm_rbf.fit(X_train_clean, y_train_clean)

svm_rbf_predictions = svm_rbf.predict(X_test)
svm_rbf_accuracy = accuracy_score(y_test, svm_rbf_predictions)
print(f'SVM RBF Kernel Accuracy: {svm_rbf_accuracy:.2f}')


# Define the MLP model
current_time = datetime.now()
print(" Starting MLP Model", current_time.strftime("%Y-%m-%d %H:%M:%S"))
mlp_model = Sequential()
mlp_model.add(Dense(32, activation='relu', input_shape=(X_train_clean.shape[1],)))
mlp_model.add(Dense(16, activation='relu'))
mlp_model.add(Dense(1, activation='sigmoid'))

mlp_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Fit the model
mlp_history = mlp_model.fit(X_train_clean, y_train_clean, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
mlp_scores = mlp_model.evaluate(X_test, y_test)
print(f'MLP Accuracy: {mlp_scores[1]:.2f}')

# OTHER ONES
# ==============================================================================
print("Other ones")

lr_model = LinearRegression()
lr_model.fit(X_train_clean, y_train_clean)

# Make predictions on the test set
lr_predictions = lr_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, lr_predictions)
mae = mean_absolute_error(y_test, lr_predictions)

print(f"Linear Regression Mean Squared Error: {mse:.2f}")
print(f"Linear Regression Mean Absolute Error: {mae:.2f}")


# Create a pipeline with the KNN classifier
knn_pipeline = make_pipeline(KNeighborsClassifier())

# Define the parameter grid for GridSearchCV
param_grid = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7],  # You can add more values to test
    'kneighborsclassifier__weights': ['uniform', 'distance'],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='accuracy')

# Fit the model to the training data
grid_search.fit(X_train_clean, y_train_clean)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print("Best Parameters:", best_params)

# Predict on the test set using the best estimator
y_pred = best_estimator.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)
