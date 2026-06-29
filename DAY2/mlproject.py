import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import os

# category_encoders is needed for Target Encoding
try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed. Target Encoding will not be available.")


def main():

    #Loading Dataser
    print("Loading Dataset...")
    file_path = "train.csv"

    if not os.path.exists(file_path):
        print(f"Error: Cannot find '{file_path}'")
        return

    df = pd.read_csv(file_path)
    print(f"Dataset Loaded Successfully")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    # Create Team_ID Column
    df["Team_ID"] = [
        "Team_" + str(np.random.randint(1, 150))
        for _ in range(len(df))
    ]

    # Target Encoding
    if TargetEncoder is not None:
        print("Applying Target Encoder...")

        encoder = TargetEncoder()

        df["Team_ID_Encoded"] = encoder.fit_transform(
            df[["Team_ID"]],
            df["H"]
        )

        print("Target Encoding Completed.\n")

    else:
        print("Category Encoder not installed.\n")

    #Features Solution
    print("Selecting Best Features...")

    features_to_test = ['R', 'AB', 'SO', 'SB']

    X_features = df[features_to_test].fillna(0)
    y_target = df["W"]      # Change if your target column is different

    selector = SelectKBest(score_func=mutual_info_regression, k=2)

    selector.fit(X_features, y_target)

    winning_features = selector.get_support()

    best_features = X_features.columns[winning_features].tolist()

    print("Best Features:", best_features)


    # Best Features
    print(best_features)

    # Splitting Data
    X = df[best_features]   # Select the best feature columns
    y = df['H']             # Target column

    #   Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Print dataset sizes
    print(f"Training Data Size: {X_train.shape}")
    print(f"Testing Data Size: {X_test.shape}")


    #Training Model
    model=LinearRegression()
    model.fit(X_train,y_train)

    prediction=model.predict(X_test)
    print(prediction)

   
    # Handling Missing Values
    print("Creating Artificial Missing Values...")

    df.loc[0:24, "H"] = np.nan

    imputer = SimpleImputer(strategy="median")
    df["H"] = imputer.fit_transform(df[["H"]]).ravel()

    print(f"Missing values in H: {df['H'].isnull().sum()}\n")

    # Log Transformation
    print("Applying Log Transformation on Runs (R)...")

    df["LogRuns"] = np.log1p(df["R"])

    print(f"New Skewness: {df['LogRuns'].skew():.2f}\n")

    # Linear Regression Example
    X = df[["H"]]
    y = df["R"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Linear Regression Model Trained Successfully.")


if __name__ == "__main__":
    main()

    