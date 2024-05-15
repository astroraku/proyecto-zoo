import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cargar el dataset
data = pd.read_csv('zoo3.csv')

# Preparar los datos
X = data.drop(['animal_name', 'class_type'], axis=1)  # Valores de entrada
y = data['class_type']  # Valores de salida

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regresión Logística
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Precisión de Regresión Logística: {accuracy_log}")

# K-Vecinos Cercanos
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Precisión de K-Vecinos Cercanos: {accuracy_knn}")

# Máquinas Vector Soporte
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Precisión de Máquinas Vector Soporte: {accuracy_svm}")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Precisión de Naive Bayes: {accuracy_nb}")

# Red Neuronal (IA)
mlp = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=20)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"Precisión de Red Neuronal: {accuracy_mlp}")
print("\n")

# Función para calcular y mostrar las métricas
def evaluate_model(model_name, y_true, y_pred):
    print(f"Resultados para {model_name}:")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("")

# Evaluar cada modelo
evaluate_model("Regresión Logística", y_test, y_pred_log)
evaluate_model("K-Vecinos Cercanos", y_test, y_pred_knn)
evaluate_model("Máquinas Vector Soporte", y_test, y_pred_svm)
evaluate_model("Naive Bayes", y_test, y_pred_nb)
evaluate_model("Red Neuronal", y_test, y_pred_mlp)

