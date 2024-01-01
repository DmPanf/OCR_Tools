from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Генерация синтетических данных
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание ансамбля с использованием Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучение модели
rf.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = rf.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Важно отметить, что этот код предназначен для иллюстрации и использует синтетические данные.
# Для реальных приложений вам потребуется загрузить и предобработать реальные данные.
