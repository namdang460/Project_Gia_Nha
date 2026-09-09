import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Doc du lieu
data = pd.read_csv("Housing.csv")

print("5 dong dau cua du lieu:")
print(data.head())

# 2. Tach bien dau vao X va bien can du doan y
X = data.drop("price", axis=1)
y = data["price"]

# 3. Xac dinh cac cot kieu chu va cot kieu so
categorical_columns = X.select_dtypes(include=["object"]).columns
numeric_columns = X.select_dtypes(exclude=["object"]).columns

# 4. Ma hoa du lieu kieu chu bang One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)

# 5. Tao mo hinh hoi quy tuyen tinh
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# 6. Chia du lieu thanh 80% train va 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 7. Huan luyen mo hinh
model.fit(X_train, y_train)

# 8. Du doan tren tap test
y_pred = model.predict(X_test)

# 9. Danh gia mo hinh
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== KET QUA DANH GIA MO HINH =====")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 : {r2:.4f}")

# 10. So sanh gia thuc te va gia du doan
result = pd.DataFrame({
    "Gia_thuc_te": y_test.values,
    "Gia_du_doan": y_pred
})

print("\n===== 10 KET QUA DAU TIEN =====")
print(result.head(10))
