# 🎌 AnimeScore-ML — Predicting Anime Success with Machine Learning

> **CSE427 · Section 03 · Group 04 · BRAC University · Spring 2026**

A comparative machine learning study that predicts MyAnimeList audience scores from anime metadata using four classical regression models and three neural network architectures. The project also investigates how genre combinations influence ratings through feature-level importance analysis.

📄 **[Read the Paper (IEEE Format)](CSE427_IEEE_FINAL_Report.pdf)**  &nbsp;|&nbsp; 🎥 **[Watch the Presentation](https://www.youtube.com/watch?v=qkZ3fk8aMvc)**  &nbsp;|&nbsp; 📦 **[Dataset on Kaggle](https://www.kaggle.com/datasets/azathoth42/myanimelist/data)**

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Research Questions](#-research-questions)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#methodology)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Feature Engineering](#feature-engineering)
  - [Models](#models)
  - [Evaluation Metrics](#evaluation-metrics)
- [Results](#-results)
  - [All 7 Models](#all-7-models-sorted-by-test-rmse)
  - [Key Findings](#key-findings)
  - [Feature Importance](#feature-importance)
- [Visualisations](#-visualisations)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)
- [Contributors](#-contributors)
- [Citation](#-citation)

---

## 🔍 Project Overview

The global anime industry releases hundreds of titles every year, yet no systematic tool exists to estimate a title's audience reception before it airs. **MyAnimeList (MAL)** aggregates rich metadata per title — genre, episode count, source material, age rating, duration, and user scores — but this data is rarely exploited for predictive analysis.

This project builds and evaluates **seven regression models** on the MyAnimeList dataset to answer two concrete questions, producing a benchmark for future pre-release score prediction systems.

---

## ❓ Research Questions

1. **Can anime success (audience score) be predicted from metadata available before a title is released?**
2. **How do genre combinations influence audience ratings, and which genres carry the most predictive signal?**

---

## 📦 Dataset

| Field | Value |
|---|---|
| **Name** | MyAnimeList Dataset |
| **Source** | [Kaggle — azathoth42/myanimelist](https://www.kaggle.com/datasets/azathoth42/myanimelist/data) |
| **Size** | 14,478 anime titles × 30 columns |
| **Target variable** | `score` (continuous, 1–10 scale) |
| **Format** | CSV (`AnimeList.csv`) |

### Key Attributes

| Column | Description |
|---|---|
| `type` | TV, Movie, OVA, ONA, Special, Music |
| `episodes` | Total episode count |
| `status` | Finished Airing / Currently Airing / Not yet aired |
| `source` | Manga, Light Novel, Original, Game, Visual Novel, etc. |
| `genre` | Comma-separated multi-label genres |
| `duration` | Average episode length (string, e.g. "24 min. per ep.") |
| `rating` | Age classification (G, PG, PG-13, R, R+, Rx) |
| `score` | ⭐ **Target** — average user rating on MAL (1–10) |
| `scored_by` | Number of users who rated the title |
| `rank` | Global ranking by score |
| `popularity` | Ranking by number of list additions |
| `members` | Users who added the title to their list |
| `favorites` | Users who marked it as a favourite |

> **Score distribution:** Bell-shaped, centred at 6.5–7.5. Very few titles score below 4 or above 9, making exact prediction non-trivial for linear models.

---

## 📁 Project Structure

```
AnimeScore-ML/
│
├── AnimeScoreML_Modeling_and_Evaluation.ipynb   ← Main notebook (EDA + Preprocessing + All 7 Models)
├── AnimeScoreML_Predicting_Anime_Success_with_Machine_Learning.pdf    ← Full IEEE-format research paper
└── README.md
```

---

## ⚙️ Methodology

### Preprocessing Pipeline

The pipeline consists of **7 ordered steps** designed to prevent data leakage and type errors.

```
Raw Dataset (14,478 × 30)
        │
        ▼
Step 1: Drop irrelevant columns (IDs, text, dates)
        │
        ▼
Step 2: Feature Engineering
        │  ├─ episode_category  (binned: short/standard/medium/long/very long)
        │  └─ duration_min      (parsed from string → numeric minutes)
        ▼
Step 3: Impute missing values
        │  ├─ Numerical  → median
        │  └─ Categorical → mode
        ▼
Step 4: Multi-label genre encoding (top 10 genres → binary columns)
        │
        ▼
Step 5: One-hot encode (type, source, rating, status, episode_category)
        │  └─ drop_first=True to avoid multicollinearity
        ▼
Step 6: Train / Test split  (80% train = 11,582 | 20% test = 2,896, seed=42)
        │
        ▼
Step 7: StandardScaler (fit on train only → transform both)
        └─ Target score is NEVER scaled
```

### Feature Engineering

| Feature | Source Column | Transformation |
|---|---|---|
| `episode_category` | `episodes` | Binned: ≤12 / 13–24 / 25–50 / 51–200 / >200 |
| `duration_min` | `duration` | Regex extract of numeric minutes from string |

After all encoding steps the final feature matrix has **48 columns**.

### Models

#### Classical Baselines

| # | Model | Key Setting | Feature Input |
|---|---|---|---|
| 1 | **Linear Regression** | — | Scaled |
| 2 | **Decision Tree** | `max_depth=10` | Unscaled |
| 3 | **Random Forest** | `n_estimators=100` | Unscaled |
| 4 | **XGBoost** | `n_estimators=200`, `lr=0.1`, `max_depth=6` | Unscaled |

#### Neural Networks (TensorFlow / Keras — trained on Colab T4 GPU)

| # | Model | Architecture | Regularisation | Stopped at |
|---|---|---|---|---|
| 5 | **Shallow NN** | Input→128→64→32→Output | BatchNorm + Dropout(0.3/0.2) | Epoch 36 |
| 6 | **Deep NN** | Input→256→128→64→32→16→Output | L2 + Dropout(0.4/0.3/0.2) + ReduceLROnPlateau | Epoch 60 |
| 7 | **Wide & Deep NN** | Wide: raw features direct; Deep: 256→128→64; Merge→32→Output | ReduceLROnPlateau | Epoch 46 |

> **Wide & Deep design rationale:** The Wide path memorises linear feature–score relationships directly; the Deep path learns non-linear patterns through stacked layers. Concatenating both paths lets the final layer blend simple memorisation with complex abstraction — inspired by Cheng et al. (2016).

### Evaluation Metrics

All metrics reported on **both train and test** sets to detect overfitting.

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

---

## 📊 Results

### All 7 Models (sorted by Test RMSE)

| Model | Type | Train MSE | Test MSE | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---|---|---|---|---|---|---|
| **Random Forest** | ML | 0.0303 | **0.1940** | 0.1740 | **0.4404** | 0.9858 | **0.9115** |
| Decision Tree | ML | 0.2053 | 0.2309 | 0.4531 | 0.4805 | 0.9037 | 0.8946 |
| XGBoost | ML | 0.0628 | 0.2634 | 0.2505 | 0.5132 | 0.9706 | 0.8797 |
| **Wide & Deep NN** | NN | 0.2228 | **0.3088** | 0.4720 | **0.5557** | 0.8954 | **0.8590** |
| Deep NN | NN | 0.3517 | 0.3605 | 0.5930 | 0.6004 | 0.8349 | 0.8354 |
| Shallow NN | NN | 0.3543 | 0.3622 | 0.5952 | 0.6018 | 0.8337 | 0.8347 |
| Linear Regression | ML | 0.4108 | 0.4122 | 0.6409 | 0.6420 | 0.8073 | 0.8118 |

### Key Findings

**1. Random Forest leads overall.**
Test RMSE 0.4404 and R² 0.9115 — more than halving the test MSE of Linear Regression (0.4122 → 0.1940). Tree ensembles capture the non-linear interactions between metadata features and score that a linear model cannot.

**2. Wide & Deep NN is the best neural network.**
Test RMSE 0.5557, MSE 0.3088, R² 0.8590. Its dual-path design outperforms both the Shallow NN and Deep NN, confirming that explicitly combining linear and non-linear paths is advantageous for this tabular dataset.

**3. Neural networks generalise more consistently.**
Train/test RMSE gaps — Shallow NN: **0.0066**, Deep NN: **0.0074**, Wide & Deep NN: **0.0837** — are far tighter than Random Forest (**0.2664**) and XGBoost (**0.2627**). Dropout and BatchNormalization are more effective at controlling overfitting than depth alone.

**4. Residuals are centred near zero across all NN models.**
Residual standard deviations: Shallow 0.563, Deep 0.552, Wide & Deep **0.550**. No model shows systematic bias across the score range.

**5. Post-release features inflate performance.**
`rank`, `scored_by`, `members`, and `favorites` are the strongest predictors but are unavailable before a title airs. A genuine pre-release model must restrict inputs to announcement-time features: `genre`, `type`, `source`, `episode count`, `age rating`, `duration`. Neural networks with Dropout are better candidates for this restricted setting.

### Feature Importance

Top 12 features from Random Forest:

```
rank                   ████████████████████████████████████████ (strongest)
status_Not yet aired   ████████████████████████████████
scored_by              ████████
popularity             ████
duration_min           ████
members                ███
favorites              ███
source_Original        ██
type_Music             ██
type_ONA               ██
status_Finished Airing ██
Drama                  ██  ← highest-ranked genre feature
```

> **Genre insight:** Drama carries the most discriminative signal among genre features. Comedy and Action appear so frequently across both high- and low-rated titles that their presence provides minimal predictive information — high prevalence dilutes discriminative power.

---

## 🖼 Visualisations

The notebook produces the following figures:

| Figure | Description |
|---|---|
| Score Distribution | Bell-shaped, centred 6.5–7.5 |
| Top 10 Genres | Comedy leads with 4,500+ entries |
| Members vs Score | High popularity ≠ high score |
| Missing Value Analysis | background (>80%), licensor (≈70%) most incomplete |
| Episode Category Distribution | Most anime are Short (≤12 ep) or Standard (13–24 ep) |
| Baseline RMSE & R² | Grouped bar chart, all 4 baselines |
| NN Validation Loss Curves | Overlaid convergence for all 3 NNs |
| NN RMSE & R² Comparison | Grouped bars, train vs test |
| Actual vs Predicted + Residuals | 2×3 grid for all 3 NNs |
| Final Comparison (all 7 models) | Horizontal bar charts, sorted by test RMSE |
| Overfitting Check | Train vs test gap for all 7 models |
| Feature Importance | Top 12 RF importances |

---

## 🚀 How to Run

### Option A — Google Colab (recommended, GPU available)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `AnimeScoreML_Modeling_and_Evaluation.ipynb`
3. Set runtime: **Runtime → Change runtime type → T4 GPU**
4. Run all cells — the dataset loads automatically from Google Drive

### Option B — Local

```bash
# 1. Clone the repository
git clone https://github.com/MohammadHasibulAmin/AnimeScore-ML.git
cd AnimeScore-ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    Place AnimeList.csv in the project root, or update the file path in cell 3

# 4. Launch Jupyter
jupyter notebook AnimeScoreML_Modeling_and_Evaluation.ipynb
```

> **Note:** Neural network training cells (Sections 6–8 of the notebook) require a GPU for reasonable training times. On CPU they will still run but will be significantly slower.

---

## 📦 Dependencies

```txt
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
xgboost>=1.7
tensorflow>=2.12        # for neural network sections
keras>=2.12
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow jupyter
```

---

## 👥 Contributors

| Name | GitHub |
|---|---|
| Mohammad Hasibul Amin | [@MohammadHasibulAmin](https://github.com/MohammadHasibulAmin) |
| Fabiha Tarannum Areena | [@FabihaTarannumA](https://github.com/FabihaTarannumA) |
| Zawad Ahsan | [@ZeddhD](https://github.com/ZeddhD) |

**Supervisor:** Confidential

**Institution:** BRAC University, Dhaka, Bangladesh
**Semester:** Spring 2026

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{amin2026animescore,
  title     = {Understanding Anime Success: Predicting Popularity and Genre
               Influence Using Machine Learning},
  author    = {Amin, Mohammad Hasibul and Areena, Fabiha Tarannum and Ahsan, Zawad},
  journal   = {CSE427 Course Project, BRAC University},
  year      = {2026},
  note      = {Dataset: MyAnimeList (Kaggle, azathoth42). Code available at
               https://github.com/MohammadHasibulAmin/AnimeScore-ML}
}
```

### References

```
[1]  Reynaldi & Istiono, Asian J. Research in Computer Science, 2023
[2]  B. Zhang, Theoretical and Natural Science, 2024
[3]  B. Soni et al., RikoNet, arXiv:2106.12970, 2021
[4]  Z. Meng, Applied and Computational Engineering, 2023
[5]  S. R. Javaji & K. Sarode, arXiv:2310.04878, 2023
[6]  J. Sosa et al., arXiv:2411.03333, 2024
[7]  Y. Kang et al., arXiv:2402.10381, 2024
[8]  J. Chen, Highlights in Science, Engineering and Technology, vol. 81, 2024
[9]  Y. Saito et al., Trans. Japanese Society for AI, vol. 39, no. 6, 2024
[10] Azathoth42, MyAnimeList Dataset, Kaggle, 2018
[11] G. James et al., An Introduction to Statistical Learning, 2nd ed., Springer, 2021
```

---

<div align="center">
  <sub>Built with 🍜 and machine learning · BRAC University · Spring 2026</sub>
</div>
