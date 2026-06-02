import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("fraudTest.csv")

drop_cols = [
    'Unnamed: 0',
    'trans_date_trans_time',
    'cc_num',
    'first',
    'last',
    'street',
    'city',
    'state',
    'zip',
    'dob',
    'trans_num'
]

df = df.drop(columns=drop_cols)

le = LabelEncoder()
print(df['category'].unique())

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
print("Categories:")
print(df['category'].unique())

print("\nGender:")
print(df['gender'].unique())

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

print("Features used:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("Model Saved Successfully!")