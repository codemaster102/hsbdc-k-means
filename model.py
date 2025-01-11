import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(file_path):
    data = pd.read_csv(file_path).dropna()
    X = data[['socioeconomic_status', 'academic_performance', 'resources']]
    return StandardScaler().fit_transform(X), ['socioeconomic_status', 'academic_performance', 'resources']

def train_kmeans(X, n_clusters=3):
    return KMeans(n_clusters=n_clusters, random_state=42).fit(X)

def predict_group(kmeans, scaler, student_data):
    student_scaled = scaler.transform(pd.DataFrame([student_data]))
    return kmeans.predict(student_scaled)[0]

def predict_groups_for_students(kmeans, scaler, students_file_path):
    students_data = pd.read_csv(students_file_path)
    students_data_scaled = scaler.transform(students_data[['socioeconomic_status', 'academic_performance', 'resources']])
    students_data['group'] = kmeans.predict(students_data_scaled)
    return students_data

if __name__ == "__main__":
    file_path = "student_data.csv"  # Path to your training data
    X_scaled, feature_columns = preprocess_data(file_path)
    kmeans_model = train_kmeans(X_scaled, n_clusters=3)
    scaler = StandardScaler().fit(X_scaled)

    # Predict group for a single example student
    example_student = {'socioeconomic_status': 0.5, 'academic_performance': 0.8, 'resources': 0.6}
    group = predict_group(kmeans_model, scaler, example_student)
    print(f"The example student belongs to group {group}.")

    # Predict groups for multiple students from a file
    students_file_path = "new_students.csv"  # Path to the new students data
    students_with_groups = predict_groups_for_students(kmeans_model, scaler, students_file_path)
    print(students_with_groups)
