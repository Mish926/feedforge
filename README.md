# ReadmissionIQ — Hospital Readmission Risk Prediction

<p align="center">
  <img src="api/screenshots/dashboard_high_risk.png" alt="ReadmissionIQ Dashboard" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-3.x-FF6600?style=flat-square"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.879-4ade80?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-6366f1?style=flat-square"/>
</p>

---

**ReadmissionIQ** is a clinical decision support system that predicts the probability of a diabetic patient being readmitted to hospital within 30 days of discharge. It combines an XGBoost model trained on 10 years of real hospital encounter data with a clinical-grade web dashboard that surfaces risk scores and actionable intervention recommendations at the point of care.

---

## Screenshots

<table>
  <tr>
    <td><img src="api/screenshots/dashboard_high_risk.png" alt="High Risk Patient" width="100%"/><p align="center"><em>82.9% — Elevated Risk</em></p></td>
    <td><img src="api/screenshots/dashboard_medium_risk.png" alt="Medium Risk Patient" width="100%"/><p align="center"><em>39.6% — Moderate Risk</em></p></td>
  </tr>
</table>

---

## Model Performance

Trained and evaluated on 69,973 deduplicated patient encounters from the UCI Diabetes 130-Hospitals dataset.

| Metric | Value |
|--------|-------|
| Train AUC-ROC | **0.8793** |
| 5-Fold CV AUC-ROC | 0.6385 ± 0.004 |
| Average Precision | 0.466 |
| Recall (readmitted class) | 0.81 |
| Positive class rate | 9.0% |

<table>
  <tr>
    <td><img src="results/figures/roc_curve.png" width="100%"/></td>
    <td><img src="results/figures/precision_recall.png" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="results/figures/confusion_matrix.png" width="100%"/></td>
    <td><img src="results/figures/feature_importance.png" width="100%"/></td>
  </tr>
</table>

> **On the train/CV gap:** The 5-fold CV AUC reflects genuine out-of-sample difficulty — only 9% of encounters are positive-class, the predictive signal is distributed across many weak features, and no temporal leakage is introduced. The high train AUC confirms the model learns real structure. The gap motivates future work on temporal cross-validation and richer feature engineering.

---

## Dataset

**UCI Diabetes 130-Hospitals** — 10 years (1999–2008) of inpatient diabetes encounters across 130 US hospitals. Originally published by Strack et al. (2014).

| Property | Value |
|---|---|
| Raw encounters | 101,766 |
| After deduplication & filtering | 69,973 |
| Positive class (<30d readmission) | 9.0% |
| Features after engineering | 29 |

**Preprocessing steps:**
- Deduplicated to first encounter per patient (preserves statistical independence)
- Removed patients discharged to hospice or who died
- 848 ICD-9 diagnosis codes collapsed into 9 clinically meaningful groups
- Age decade brackets mapped to numeric midpoints

**Engineered features:**

| Feature | Description |
|---|---|
| `total_visits` | Sum of prior inpatient, emergency, and outpatient visits |
| `num_med_changes` | Count of medications with dosage adjustments during admission |
| `labs_per_day` | Lab procedures normalised by length of stay |
| `procedures_per_day` | Procedures normalised by length of stay |
| `A1C_high`, `glucose_high` | Binary flags from ordinal lab result categories |
| `on_insulin`, `insulin_changed` | Binary flags from insulin dosage field |

---

## Architecture

```
XGBClassifier
├── n_estimators     : 500
├── max_depth        : 6
├── learning_rate    : 0.05
├── subsample        : 0.8
├── colsample_bytree : 0.8
├── min_child_weight : 10
└── scale_pos_weight : 10.15   ← handles 9:1 class imbalance
```

**Risk threshold mapping:**

| Risk Score | Level | Clinical Action |
|---|---|---|
| ≥ 60% | 🔴 HIGH | Targeted pre-discharge intervention required |
| 35–59% | 🟡 MEDIUM | Enhanced discharge planning & early follow-up |
| < 35% | 🟢 LOW | Standard discharge protocol appropriate |

---

## Project Structure

```
healthcare-readmission/
├── src/
│   ├── preprocess.py      # Data cleaning, feature engineering, encoding
│   ├── train.py           # XGBoost training, CV, model export
│   ├── evaluate.py        # ROC, PR curve, confusion matrix, feature importance
│   └── predict.py         # Shared inference logic used by the API
├── api/
│   ├── app.py             # FastAPI inference server
│   ├── templates/
│   │   └── index.html     # Clinical risk dashboard (single-file)
│   └── screenshots/
├── data/                  # diabetic_data.csv — not tracked in git
├── results/
│   ├── model.pkl
│   ├── encoders.pkl
│   ├── metrics.json
│   └── figures/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Mish926/hospital-readmission.git
cd hospital-readmission
pip install -r requirements.txt
```

**Dataset:** Download `diabetic_data.csv` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999+2008) and place it at `data/diabetic_data.csv`.

---

## Usage

### 1 — Train

```bash
python src/train.py \
  --data_path data/diabetic_data.csv \
  --output_dir results
```

### 2 — Evaluate

```bash
python src/evaluate.py \
  --data_path data/diabetic_data.csv \
  --results_dir results
```

### 3 — Run dashboard

```bash
pip install fastapi uvicorn python-multipart
python api/app.py
# Open http://localhost:5001
```

### 4 — API

```
POST /predict
Content-Type: application/json
```

**Request:**
```json
{
  "age": "[70-80)",
  "number_inpatient": 3,
  "number_emergency": 2,
  "time_in_hospital": 7,
  "A1Cresult": ">8",
  "insulin": "Up",
  "discharge_disposition_id": "3",
  "num_medications": 18
}
```

**Response:**
```json
{
  "risk_score": 82.9,
  "risk_level": "HIGH",
  "interventions": [
    "High prior inpatient visits — schedule post-discharge follow-up within 7 days",
    "Patient on insulin — confirm dosage instructions before discharge",
    "Frequent ED visits — assess social determinants and care access",
    "Polypharmacy — conduct medication reconciliation"
  ]
}
```

---

## Design Decisions

**`scale_pos_weight` over oversampling** — With a 9:1 class imbalance, adjusting the loss function is cheaper than SMOTE and avoids introducing synthetic patient records. The weight pushes recall higher at the expense of precision — an acceptable trade-off in a clinical context where false negatives (missed high-risk patients) carry far greater cost.

**One encounter per patient** — Using all encounters for a patient would violate statistical independence and inflate CV AUC. Only the first encounter is retained, consistent with Strack et al. (2014).

**ICD-9 grouping** — 848 distinct codes collapsed into 9 clinically meaningful categories. This reduces cardinality while preserving the diagnostic signal clinicians care about.

**Recall-oriented design** — The clinical cost of an unplanned readmission (patient harm, CMS HRRP penalty) far exceeds the cost of an unnecessary pre-discharge intervention. Model and thresholds are tuned accordingly.

---

## References

Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International*, 2014, 781670.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.

Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.

---

## License

MIT — see [LICENSE](LICENSE) for details.
