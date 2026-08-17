from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# STUDENT ACADEMIC ADVISOR
# AI Skills Course Project
# REAL DATASET: UCI Student Performance - student-mat.csv
# ============================================================

DATA_FILE = "student-mat.csv"
OUTPUT_DIR = Path("outputs")


# ============================================================
# 1. LOAD REAL DATA
# ============================================================

def load_data() -> pd.DataFrame:
    path = Path(DATA_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. Put student-mat.csv in the same "
            "folder as this Python file."
        )

    # UCI student-mat.csv is semicolon-separated.
    df = pd.read_csv(path, sep=";")

    if df.empty:
        raise ValueError("The dataset is empty.")

    required = ["studytime", "absences", "G1", "G2", "G3"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ============================================================
# 2. DOMAIN FEATURES
# ============================================================

def build_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The real UCI dataset does not contain GPA, Attendance or Workload
    columns directly. Therefore we derive project features transparently:

    GPA / academic performance:
        average of G1, G2, G3 converted from 0-20 to 0-4.

    Attendance:
        100% minus a normalized absence penalty.

    Workload:
        studytime from the original UCI dataset, where larger values
        represent more weekly study time.

    No synthetic student records are created.
    """
    out = pd.DataFrame(index=df.index)

    out["GPA"] = (
        pd.to_numeric(df["G1"], errors="coerce")
        + pd.to_numeric(df["G2"], errors="coerce")
        + pd.to_numeric(df["G3"], errors="coerce")
    ) / 3 / 5

    absences = pd.to_numeric(df["absences"], errors="coerce")

    # Convert absence count to an interpretable attendance percentage.
    # The maximum observed absence is used only for normalization.
    max_absences = max(float(absences.max()), 1.0)
    out["Attendance"] = 100 - (absences / max_absences * 60)

    # studytime is ordinal in the UCI data:
    # 1=<2h, 2=2-5h, 3=5-10h, 4=>10h per week.
    out["Workload"] = pd.to_numeric(df["studytime"], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))

    return out.reset_index(drop=True)


def normalize_domain_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()

    out["gpa_norm"] = np.clip(out["GPA"] / 4.0, 0, 1)
    out["attendance_norm"] = np.clip(out["Attendance"] / 100.0, 0, 1)

    # studytime has values 1..4.
    out["workload_norm"] = np.clip((out["Workload"] - 1) / 3, 0, 1)

    return out


# ============================================================
# 3. KNOWLEDGE REPRESENTATION
# ============================================================

@dataclass
class OAVFact:
    object_name: str
    attribute: str
    value: float


@dataclass
class SemanticNode:
    name: str
    parent: str | None = None
    relation: str = "is-a"


def build_knowledge_representation(facts: Dict[str, float]):
    # O-A-V triplets
    oav = [
        OAVFact("Student", "GPA", facts["gpa"]),
        OAVFact("Student", "Attendance", facts["attendance"]),
        OAVFact("Student", "Workload", facts["workload"]),
    ]

    # Hierarchical / semantic network
    semantic = [
        SemanticNode("Student"),
        SemanticNode("AcademicPerformance", "Student"),
        SemanticNode("AttendanceBehavior", "Student"),
        SemanticNode("WorkloadLevel", "Student"),
        SemanticNode("AcademicRisk", "Student"),
        SemanticNode("Intervention", "Student"),
        SemanticNode("HighRisk", "AcademicRisk"),
        SemanticNode("ModerateRisk", "AcademicRisk"),
        SemanticNode("LowRisk", "AcademicRisk"),
        SemanticNode("UrgentIntervention", "Intervention"),
        SemanticNode("NormalIntervention", "Intervention"),
    ]

    return oav, semantic


# ============================================================
# 4. FUZZY LOGIC
# ============================================================

def AND(*values: float) -> float:
    return min(values)


def OR(*values: float) -> float:
    return max(values)


def NOT(value: float) -> float:
    return 1.0 - value


def left_shoulder(x: float, a: float, b: float) -> float:
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def triangular(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def right_shoulder(x: float, a: float, b: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def fuzzy_memberships(gpa, attendance, workload):
    return {
        "gpa_low": left_shoulder(gpa, 0.25, 0.50),
        "gpa_medium": triangular(gpa, 0.35, 0.60, 0.80),
        "gpa_high": right_shoulder(gpa, 0.65, 0.85),

        "attendance_low": left_shoulder(attendance, 0.45, 0.70),
        "attendance_medium": triangular(attendance, 0.55, 0.75, 0.90),
        "attendance_high": right_shoulder(attendance, 0.80, 0.95),

        "workload_low": left_shoulder(workload, 0.25, 0.50),
        "workload_medium": triangular(workload, 0.35, 0.60, 0.80),
        "workload_high": right_shoulder(workload, 0.65, 0.85),
    }


# ============================================================
# 5. PRODUCTION RULES
# ============================================================

@dataclass
class Rule:
    rule_id: str
    description: str
    conclusion: str
    premise: Callable[[Dict[str, float]], float]
    dependencies: Set[str]
    cf: float
    conclusion_type: str


def build_rules():
    return [
        Rule(
            "R1",
            "IF GPA is low AND Attendance is low THEN AcademicRiskHigh",
            "AcademicRiskHigh",
            lambda f: AND(f["gpa_low"], f["attendance_low"]),
            set(),
            0.95,
            "intermediate",
        ),

        Rule(
            "R2",
            "IF GPA is low AND (Attendance is low OR Workload is high) THEN AcademicRiskHigh",
            "AcademicRiskHigh",
            lambda f: AND(
                f["gpa_low"],
                OR(f["attendance_low"], f["workload_high"])
            ),
            set(),
            0.90,
            "intermediate",
        ),

        Rule(
            "R3",
            "IF GPA is medium AND Attendance is low THEN AcademicRiskModerate",
            "AcademicRiskModerate",
            lambda f: AND(f["gpa_medium"], f["attendance_low"]),
            set(),
            0.85,
            "intermediate",
        ),

        Rule(
            "R4",
            "IF GPA is high AND Attendance is high THEN AcademicRiskLow",
            "AcademicRiskLow",
            lambda f: AND(f["gpa_high"], f["attendance_high"]),
            set(),
            0.95,
            "intermediate",
        ),

        Rule(
            "R5",
            "IF AcademicRiskHigh THEN UrgentIntervention",
            "UrgentIntervention",
            lambda f: f.get("AcademicRiskHigh", 0.0),
            {"AcademicRiskHigh"},
            0.95,
            "final",
        ),

        Rule(
            "R6",
            "IF AcademicRiskModerate AND Workload is high THEN UrgentIntervention",
            "UrgentIntervention",
            lambda f: AND(
                f.get("AcademicRiskModerate", 0.0),
                f["workload_high"]
            ),
            {"AcademicRiskModerate"},
            0.90,
            "final",
        ),

        Rule(
            "R7",
            "IF AcademicRiskModerate THEN NormalIntervention",
            "NormalIntervention",
            lambda f: f.get("AcademicRiskModerate", 0.0),
            {"AcademicRiskModerate"},
            0.85,
            "final",
        ),

        Rule(
            "R8",
            "IF AcademicRiskLow THEN NormalIntervention",
            "NormalIntervention",
            lambda f: f.get("AcademicRiskLow", 0.0),
            {"AcademicRiskLow"},
            0.75,
            "final",
        ),
    ]


# ============================================================
# 6. DEPENDENCY RESOLUTION + FORWARD CHAINING
# ============================================================

def topological_order(rules):
    by_conclusion = {r.conclusion: r for r in rules}

    graph = {r.rule_id: set() for r in rules}
    indegree = {r.rule_id: 0 for r in rules}

    for rule in rules:
        for dependency in rule.dependencies:
            if dependency in by_conclusion:
                parent = by_conclusion[dependency]
                if rule.rule_id not in graph[parent.rule_id]:
                    graph[parent.rule_id].add(rule.rule_id)
                    indegree[rule.rule_id] += 1

    queue = [r.rule_id for r in rules if indegree[r.rule_id] == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(current)

        for nxt in graph[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(result) != len(rules):
        raise ValueError("Circular dependency detected in rule base.")

    by_id = {r.rule_id: r for r in rules}
    return [by_id[x] for x in result]


def forward_chain(facts, rules):
    working = dict(facts)
    trace = []

    for rule in topological_order(rules):
        fv = float(np.clip(rule.premise(working), 0, 1))
        cv = fv * rule.cf
        fired = cv > 0

        if fired:
            working[rule.conclusion] = max(
                working.get(rule.conclusion, 0.0),
                cv
            )

        trace.append({
            "Rule": rule.rule_id,
            "Description": rule.description,
            "FV": round(fv, 4),
            "CF": rule.cf,
            "CV": round(cv, 4),
            "Fired": fired,
            "Conclusion": rule.conclusion,
            "Conclusion_Type": rule.conclusion_type,
        })

    return working, trace


def union_method(values):
    result = 0.0

    for value in values:
        result = result + value - result * value

    return result


def aggregate_final_conclusions(facts):
    urgent = facts.get("UrgentIntervention", 0.0)
    normal = facts.get("NormalIntervention", 0.0)

    return {
        "UrgentIntervention_CV": urgent,
        "NormalIntervention_CV": normal,
        "Maximum_Method": max(urgent, normal),
        "Union_Method": union_method([urgent, normal]),
    }


# ============================================================
# 7. ANALYZE ONE STUDENT
# ============================================================

def analyze_student(normalized, student_number):
    row = normalized.iloc[student_number - 1]

    fuzzy_inputs = {
        "gpa": float(row["gpa_norm"]),
        "attendance": float(row["attendance_norm"]),
        "workload": float(row["workload_norm"]),
    }

    memberships = fuzzy_memberships(
        fuzzy_inputs["gpa"],
        fuzzy_inputs["attendance"],
        fuzzy_inputs["workload"],
    )

    facts = {**fuzzy_inputs, **memberships}

    final_facts, trace = forward_chain(
        facts,
        build_rules()
    )

    aggregation = aggregate_final_conclusions(final_facts)

    return row, fuzzy_inputs, memberships, trace, aggregation


def print_student_analysis(
    row,
    student_number,
    fuzzy_inputs,
    memberships,
    trace,
    aggregation,
):
    print("\n" + "=" * 75)
    print("SELECTED STUDENT — EXPERT SYSTEM")
    print("=" * 75)

    print(f"Student number : {student_number}")
    print(f"GPA            : {row['GPA']:.2f} / 4")
    print(f"Attendance     : {row['Attendance']:.2f}%")
    print(f"Studytime      : {row['Workload']:.0f} (UCI studytime 1-4)")

    print("\nFuzzy input values:")
    for k, v in fuzzy_inputs.items():
        print(f"  {k:<18} = {v:.3f}")

    print("\nMembership values:")
    for k, v in memberships.items():
        print(f"  {k:<22} = {v:.3f}")

    print("\nInference / Rule Trace:")
    print("-" * 75)

    for x in trace:
        status = "FIRED" if x["Fired"] else "not fired"

        print(
            f"{x['Rule']} | {status:<10} | "
            f"FV={x['FV']:.3f} | "
            f"CF={x['CF']:.2f} | "
            f"CV={x['CV']:.3f} | "
            f"{x['Conclusion']}"
        )

    print("\nFinal conclusions:")
    print(
        f"  Urgent Intervention CV : "
        f"{aggregation['UrgentIntervention_CV']:.3f}"
    )
    print(
        f"  Normal Intervention CV : "
        f"{aggregation['NormalIntervention_CV']:.3f}"
    )
    print(
        f"  Maximum Method         : "
        f"{aggregation['Maximum_Method']:.3f}"
    )
    print(
        f"  Union Method           : "
        f"{aggregation['Union_Method']:.3f}"
    )

    if aggregation["UrgentIntervention_CV"] >= aggregation["NormalIntervention_CV"]:
        print("\nFINAL DECISION: Urgent academic intervention")
    else:
        print("\nFINAL DECISION: Normal academic monitoring")


# ============================================================
# 8. INFERENCE NETWORK DIAGRAM
# ============================================================

def create_inference_network():
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def square(x, y, text, intermediate=False):
        rect = plt.Rectangle(
            (x - 0.9, y - 0.35),
            1.8,
            0.7,
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rect)

        if intermediate:
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    0.18,
                    fill=False,
                    linewidth=2,
                )
            )

        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9,
        )

    def circle(x, y, text):
        ax.add_patch(
            plt.Circle(
                (x, y),
                0.6,
                fill=False,
                linewidth=2,
            )
        )
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8,
        )

    def gate(x, y, text):
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fill=False,
                linewidth=1.5,
            ),
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=1.5),
        )

    square(2, 7.5, "GPA Low")
    square(2, 5.5, "Attendance Low")
    square(2, 3.5, "Workload High")

    gate(4, 6.5, "AND")
    gate(4, 4.5, "OR")

    square(6.2, 6.5, "AcademicRiskHigh", True)
    square(6.2, 3.5, "AcademicRiskModerate", True)

    gate(8.5, 6.0, "AND")

    circle(10.8, 6.0, "Urgent\nIntervention")
    circle(10.8, 3.5, "Normal\nIntervention")

    arrow(2.9, 7.5, 3.5, 6.7)
    arrow(2.9, 5.5, 3.5, 6.3)
    arrow(4.5, 6.5, 5.25, 6.5)

    arrow(2.9, 5.5, 3.5, 4.6)
    arrow(2.9, 3.5, 3.5, 4.4)
    arrow(4.5, 4.5, 5.25, 3.7)

    arrow(7.1, 6.5, 7.9, 6.0)
    arrow(2.9, 3.5, 7.9, 5.7)
    arrow(9.1, 6.0, 10.1, 6.0)

    arrow(7.1, 3.5, 10.1, 3.5)

    ax.text(
        7,
        8.5,
        "Student Academic Advisor — Inference Network",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )

    ax.text(
        7,
        0.5,
        "Square = fact | Square + circle = intermediate | Circle = final conclusion",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "inference_network.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 9. K-MEANS — REAL DATASET
# ============================================================

def prepare_ml_data(df):
    """
    Use the complete real student dataset.
    Missing values -> imputation.
    Categorical features -> one-hot encoding.
    Numeric features -> StandardScaler.
    G3 is excluded from clustering to avoid using the final grade as
    a direct target-like variable.
    """
    data = df.copy()

    # Exclude final grade from clustering; keep G1 and G2 as earlier
    # academic-performance indicators.
    data = data.drop(columns=["G3"], errors="ignore")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    X = preprocessor.fit_transform(data)

    return np.asarray(X, dtype=float), preprocessor


def select_best_k(X):
    rows = []

    max_k = min(8, len(X) - 1)

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = model.fit_predict(X)

        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(X, labels),
        })

    scores = pd.DataFrame(rows)

    best_k = int(
        scores.loc[
            scores["silhouette_score"].idxmax(),
            "k"
        ]
    )

    return best_k, scores


def run_kmeans(df, domain_features):
    X, _ = prepare_ml_data(df)

    best_k, scores = select_best_k(X)

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20,
    )

    labels = model.fit_predict(X)

    # Elbow plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        scores["k"],
        scores["inertia"],
        marker="o",
    )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "elbow_method.png",
        dpi=180,
    )
    plt.close(fig)

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        scores["k"],
        scores["silhouette_score"],
        marker="o",
    )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "silhouette_scores.png",
        dpi=180,
    )
    plt.close(fig)

    # 2D visualization using first two transformed features.
    x1 = X[:, 0]
    x2 = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        x1,
        x2,
        c=labels,
        alpha=0.75,
    )
    ax.set_xlabel("Transformed Feature 1")
    ax.set_ylabel("Transformed Feature 2")
    ax.set_title("K-Means Student Clusters")
    fig.colorbar(
        scatter,
        ax=ax,
        label="Cluster",
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "kmeans_clusters.png",
        dpi=180,
    )
    plt.close(fig)

    clustered = domain_features.copy()
    clustered["Cluster"] = labels

    summary = (
        clustered
        .groupby("Cluster")[["GPA", "Attendance", "Workload"]]
        .mean()
        .round(2)
        .reset_index()
    )

    overall = summary[
        ["GPA", "Attendance", "Workload"]
    ].mean()

    interpretations = []

    for _, row in summary.iterrows():
        if (
            row["GPA"] < overall["GPA"]
            and row["Attendance"] < overall["Attendance"]
        ):
            interpretation = "At-risk / needs academic support"
        elif (
            row["GPA"] >= overall["GPA"]
            and row["Attendance"] >= overall["Attendance"]
        ):
            interpretation = "Higher-performing / stable"
        else:
            interpretation = "Mixed-performance / monitor"

        interpretations.append(interpretation)

    summary["Interpretation"] = interpretations

    return X, best_k, scores, labels, summary


# ============================================================
# 10. SAVE OUTPUTS
# ============================================================

def save_outputs(
    df,
    domain,
    student_number,
    fuzzy_inputs,
    memberships,
    trace,
    aggregation,
    scores,
    labels,
    summary,
):
    OUTPUT_DIR.mkdir(exist_ok=True)

    pd.DataFrame(trace).to_csv(
        OUTPUT_DIR / "expert_system_rule_trace.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {"Membership": k, "Value": v}
            for k, v in memberships.items()
        ]
    ).to_csv(
        OUTPUT_DIR / "fuzzy_memberships.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {"Metric": k, "Value": v}
            for k, v in aggregation.items()
        ]
    ).to_csv(
        OUTPUT_DIR / "final_aggregation.csv",
        index=False,
    )

    pd.DataFrame(
        [{
            "Student_Number": student_number,
            "GPA_Fuzzy": fuzzy_inputs["gpa"],
            "Attendance_Fuzzy": fuzzy_inputs["attendance"],
            "Workload_Fuzzy": fuzzy_inputs["workload"],
        }]
    ).to_csv(
        OUTPUT_DIR / "selected_student.csv",
        index=False,
    )

    scores.to_csv(
        OUTPUT_DIR / "kmeans_model_selection.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "cluster_summary.csv",
        index=False,
    )

    labeled = domain.copy()
    labeled["Cluster"] = labels

    labeled.to_csv(
        OUTPUT_DIR / "students_with_clusters.csv",
        index=False,
    )

    df.describe(include="all").T.to_csv(
        OUTPUT_DIR / "dataset_description.csv"
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():
    # Create the output folder before any CSV/image is written.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STUDENT ACADEMIC ADVISOR")
    print("AI Skills Course Project")
    print("UCI Student Performance — REAL DATASET")
    print("=" * 75)

    df = load_data()

    print(f"\nDataset loaded successfully.")
    print(f"Students : {len(df)}")
    print(f"Features : {len(df.columns)}")

    domain = build_domain_features(df)
    normalized = normalize_domain_features(domain)

    # ---------------- PART A ----------------
    print("\n" + "=" * 75)
    print("PART A — EXPERT SYSTEM")
    print("=" * 75)

    while True:
        choice = input(
            f"\nEnter student number (1-{len(df)}): "
        ).strip()

        try:
            student_number = int(choice)

            if 1 <= student_number <= len(df):
                break

        except ValueError:
            pass

        print(
            f"Invalid number. Enter an integer from 1 to {len(df)}."
        )

    row, fuzzy_inputs, memberships, trace, aggregation = analyze_student(
        normalized,
        student_number,
    )

    print_student_analysis(
        row,
        student_number,
        fuzzy_inputs,
        memberships,
        trace,
        aggregation,
    )

    # Knowledge representation files
    oav, semantic = build_knowledge_representation(fuzzy_inputs)

    pd.DataFrame([
        {
            "Object": x.object_name,
            "Attribute": x.attribute,
            "Value": x.value,
        }
        for x in oav
    ]).to_csv(
        OUTPUT_DIR / "knowledge_oav.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "Concept": x.name,
            "Parent": x.parent,
            "Relation": x.relation,
        }
        for x in semantic
    ]).to_csv(
        OUTPUT_DIR / "knowledge_semantic_network.csv",
        index=False,
    )

    create_inference_network()

    # ---------------- PART B ----------------
    print("\n" + "=" * 75)
    print("PART B — K-MEANS CLUSTERING")
    print("=" * 75)

    X, best_k, scores, labels, summary = run_kmeans(
        df,
        domain,
    )

    print(f"\nPrepared ML matrix: {X.shape}")
    print("\nK selection:")
    print(scores.to_string(index=False))
    print(f"\nSelected K = {best_k}")

    print("\nCluster interpretation:")
    print(summary.to_string(index=False))

    save_outputs(
        df,
        domain,
        student_number,
        fuzzy_inputs,
        memberships,
        trace,
        aggregation,
        scores,
        labels,
        summary,
    )

    print("\n" + "=" * 75)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
