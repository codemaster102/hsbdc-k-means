# Student Clustering with K-Means

This project uses k-means clustering to group students based on their socio-economic status, academic performance, and available resources. It predicts the cluster for a new student based on the trained model.

## Files

1. **kmeans_student_grouping.py**: The main Python script that implements k-means clustering.
2. **student_data.csv**: Sample dataset of 200 students with factors `socioeconomic_status`, `academic_performance`, and `resources`.

## Usage

### Step 1: Prepare the Environment
Ensure Python and the required libraries are installed:
```bash
pip install pandas numpy scikit-learn
```

### Step 2: Use the Sample Data
The `student_data.csv` file contains 200 randomly generated student records. You can use it directly or replace it with your dataset.

### Step 3: Run the Script
Execute the Python script to train the model and predict the cluster for a sample student:
```bash
python kmeans_student_grouping.py
```

### Step 4: Predict for New Students
Update the `example_student` dictionary in the script with new data to classify another student.

## Sample Output
After running the script, you will see output similar to:
```
The student belongs to group 2.
```

## Dataset Details
The dataset includes:
- **socioeconomic_status**: A value between 0 and 1 indicating the socio-economic background.
- **academic_performance**: A value between 0 and 1 representing academic performance.
- **resources**: A value between 0 and 1 showing access to resources.

## Customization
- Modify `n_clusters` in the script to change the number of clusters.
- Replace `student_data.csv` with a other datasets for practical use.
