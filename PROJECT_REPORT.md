# Student Academic Advisor — Project Report

## 1. Project Domain
The project develops an AI-based Student Academic Advisor. The expert system
uses academic performance, attendance behavior, and study workload as fuzzy
facts and derives academic-risk and intervention conclusions.

## 2. Dataset
The project uses the real UCI Student Performance dataset (`student-mat.csv`).

The original dataset contains student attributes including `studytime`,
`absences`, `G1`, `G2`, and `G3`.

Because the original file does not contain a literal GPA, Attendance, or
Workload column, the project uses transparent derived features:
- GPA = mean(G1,G2,G3)/5
- Attendance = 100 - normalized absence penalty
- Workload = studytime

## 3. Knowledge Representation
Two distinct representations are implemented:
1. Object-Attribute-Value (O-A-V) triplets.
2. Hierarchical/semantic network.

Production rules are stored separately from the inference engine.

## 4. Rule Base
The rule base contains:
- nested condition:
  IF GPA is low AND (Attendance is low OR Workload is high)
- chained rules where an intermediate risk conclusion becomes a premise
  for a final intervention conclusion
- intermediate conclusions:
  AcademicRiskHigh, AcademicRiskModerate, AcademicRiskLow
- final conclusions:
  UrgentIntervention, NormalIntervention

## 5. Inference Engine
Forward chaining is used.

Rule dependencies are resolved using topological ordering.
Circular dependencies are detected.

The system records:
- rule order
- FV
- CF
- CV
- fired/not fired status
- conclusion type

This provides explainability.

## 6. Fuzzy Logic
Facts are represented using membership values in [0,1].

Operators:
- AND = minimum
- OR = maximum
- NOT = 1 - value

For each rule:
FV = premise fuzzy value
CV = FV × CF

Final conclusions are aggregated using:
- Maximum Method
- Union Method U(a,b)=a+b-a*b

## 7. Machine Learning
K-Means clustering is applied to the complete student dataset.

Preprocessing:
- numeric missing values -> median imputation
- categorical missing values -> most-frequent imputation
- categorical variables -> one-hot encoding
- numeric variables -> StandardScaler

The number of clusters is evaluated using both:
- Elbow Method
- Silhouette Score

The highest silhouette score is used to select K.

## 8. Interpretation
The cluster summary compares average GPA, attendance, and workload for each
cluster. Clusters are described as higher-performing/stable,
at-risk/needs support, or mixed-performance/monitor based on their relative
domain averages.

These descriptions are interpretations of unsupervised clusters, not
ground-truth labels.
