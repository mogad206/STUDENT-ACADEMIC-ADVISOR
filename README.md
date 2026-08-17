# Student Academic Advisor — Final Version

## Dataset
This project uses the real **UCI Student Performance** dataset:
`student-mat.csv`.

Place `student-mat.csv` in the same folder as:
`student_academic_advisor.py`

The dataset is read with `sep=";"`, matching the original UCI file format.

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python student_academic_advisor.py
```

The program asks:

```text
Enter student number (1-395):
```

Enter any student number for the Expert System demonstration.

## Domain feature mapping

The UCI dataset does not contain columns literally named GPA, Attendance,
or Workload. The project derives them transparently:

- GPA = average(G1, G2, G3) converted from the 0–20 scale to a 0–4 scale.
- Attendance = 100 minus a normalized absence penalty based on `absences`.
- Workload = the original UCI `studytime` value (1–4).

No synthetic student records are used.

## Part A
- O-A-V triplets
- Semantic/hierarchical representation
- IF-THEN production rules
- Forward chaining
- Dependency/topological ordering
- Circular dependency detection
- Explainability / rule trace
- Fuzzy values [0,1]
- AND=min, OR=max, NOT=1-value
- FV, CF, CV
- Maximum aggregation
- Union aggregation
- Nested condition
- Chained rules
- Intermediate and final conclusions
- Inference network diagram

## Part B
- K-Means clustering
- Missing-value handling
- Categorical encoding
- Standard scaling
- Elbow method
- Silhouette score
- Cluster visualization
- Cluster interpretation

`G3` is excluded from K-Means feature input because it is the final grade
and can act like a target/final-outcome variable. Earlier grades and other
student features are used instead.

## Generated output
The program creates an `outputs` folder containing:
- expert_system_rule_trace.csv
- fuzzy_memberships.csv
- final_aggregation.csv
- selected_student.csv
- knowledge_oav.csv
- knowledge_semantic_network.csv
- inference_network.png
- kmeans_model_selection.csv
- elbow_method.png
- silhouette_scores.png
- kmeans_clusters.png
- cluster_summary.csv
- students_with_clusters.csv
- dataset_description.csv
