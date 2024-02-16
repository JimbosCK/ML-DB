from Printer import Printer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

class DataHelper:
    def __init__(self,show_tables=True, printer=Printer(enabled=True)):
        self.show_tables = show_tables
        self.printer = printer

    def showInitData(self, df):
        self.printer.console(df.head())
        #self.printer.console(df.info())
        self.printer.console(df.describe().transpose())
        self.printer.console(df.shape)
        self.printer.console(df.isnull().sum())
        #Checking the number of unique values
        self.printer.console(df.select_dtypes(include='int64').nunique())
        #check duplicate values
        self.printer.console(df.duplicated().sum())
        #drop the duplicated values
        df = df.drop_duplicates()
        self.printer.console(df.shape)

        column_names = df.columns.tolist()
        self.printer.console("Column Names:")
        self.printer.console(column_names)

        #Data Visualization
        if self.show_tables:
            numeric_columns = df.select_dtypes(include=['int64'])
            numeric_columns.hist(bins=20, figsize=(15, 10))
            plt.show()

            # Combined side-by-side count plot for categorical variables
            categorical_columns = ['blood_glucose_level','smoking_history',]
            fig, axes = plt.subplots(nrows=1, ncols=len(categorical_columns), figsize=(14, 5))

            for i, col in enumerate(categorical_columns):
                sns.countplot(x=col, data=df, ax=axes[i], palette='pastel')
                axes[i].set_title(f'Count Plot of {col}')

            plt.tight_layout()
            plt.show()


            #Stacked Area Chart .
            crosstab = pd.crosstab(df['age'],df['blood_glucose_level'])
            crosstab.plot(kind='area', colormap='viridis', alpha=0.7, stacked=True)
            plt.title('Stacked Area Chart: Age Category by Blood_glucose_level')
            plt.xlabel('Age Category')
            plt.ylabel('Count')
            plt.show()


            #Stacked Area Chart .
            crosstab = pd.crosstab(df['age'],df['diabetes'])
            crosstab.plot(kind='area', colormap='viridis', alpha=0.7, stacked=True)
            plt.title('Stacked Area Chart: Age Category by General Health')
            plt.xlabel('Age Category')
            plt.ylabel('Count')
            plt.show()

    def format_obj_col(self, df):
        # Create a copy of the DataFrame to avoid modifying the original
        df_encoded = df.copy()

        # Create a label encoder object
        label_encoder = LabelEncoder()

        # Iterate through each object column and encode its values
        for column in df_encoded.select_dtypes(include='object'):
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

        # Now, df_encoded contains the label-encoded categorical columns
        self.printer.console(df_encoded.head())

        #Correlation Heatmap
        if self.show_tables:
            plt.figure(figsize=(20, 16))
            test = sns.heatmap(df_encoded.corr(), fmt='.2g', annot=True)
            plt.show()

        return df_encoded

    def smote_resample(self, df):
        X = df.drop(columns=['diabetes'])  # Features
        y = df['diabetes']  # Target variable
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced

    def IQR(self, X_train, y_train):
        # Define the columns to remove outliers
        selected_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

        # Calculate the IQR for the selected columns in the training data
        Q1 = X_train[selected_columns].quantile(0.25)
        Q3 = X_train[selected_columns].quantile(0.75)
        IQR = Q3 - Q1

        # SetTING a threshold value for outlier detection (e.g., 1.5 times the IQR)
        threshold = 1.5

        # CreatING a mask for outliers in the selected columns
        outlier_mask = (
                (X_train[selected_columns] < (Q1 - threshold * IQR)) |
                (X_train[selected_columns] > (Q3 + threshold * IQR))
        ).any(axis=1)


        # Remove rows with outliers from X_train and y_train
        X_train_clean = X_train[~outlier_mask]
        y_train_clean = y_train[~outlier_mask]

        # Print the number of rows removed
        num_rows_removed = len(X_train) - len(X_train_clean)
        self.printer.console(f"Number of rows removed due to outliers: {num_rows_removed}")

        return X_train_clean, y_train_clean
