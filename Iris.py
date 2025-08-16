from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ====================
# 1. Khởi tạo Flask
# ====================
app = Flask(__name__)

# ====================
# 2. Huấn luyện model KNN
# ====================
# Đọc dữ liệu
df = pd.read_csv("Iris/Iris.csv")

# Tách dữ liệu
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train mô hình
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# ====================
# 3. Routes
# ====================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            sl = float(request.form["sepal_length"])
            sw = float(request.form["sepal_width"])
            pl = float(request.form["petal_length"])
            pw = float(request.form["petal_width"])

            sample = [[sl, sw, pl, pw]]
            pred_species = knn.predict(sample)
            result = f"🌸 Loài hoa dự đoán: {pred_species[0]}"
        except:
            result = "❌ Lỗi: Vui lòng nhập đúng số!"

    return render_template("index.html", result=result)


# ====================
# 4. Run app
# ====================
if __name__ == "__main__":
    app.run(debug=True)
