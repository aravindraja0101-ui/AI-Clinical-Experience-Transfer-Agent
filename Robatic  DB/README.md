# Robot-Assisted Minimally Invasive Surgery (RAMIS) Synthetic Dataset

## Dataset Objective
This dataset contains simulated frame-level and time-window-level observations extracted from robotic surgery training sessions. It is designed to serve as a high-fidelity synthetic benchmark for research in surgical AI, biomedical computer vision, surgical skill evaluation, and machine learning/deep learning model training.

The dataset includes physical metrics, visual qualities, kinematics, optical flow estimates, task-specific event frequencies, and deep embeddings (CNN and Transformer spatial-temporal features) to evaluate surgical competency across three classes: **Novice**, **Intermediate**, and **Expert**.

---

## Dataset Summary Statistics
* **Total Observations (Rows)**: 186,532
* **Features (Columns)**: 100
* **Skill Level Class Distribution**:
  * **Novice**: 65,286 rows (35.0%)
  * **Intermediate**: 65,286 rows (35.0%)
  * **Expert**: 55,960 rows (30.0%)
* **Missing Values**: 0
* **Duplicate Rows**: 0

---

## Generation Methodology
The dataset is generated via a hierarchical mathematical model:
1. **Trial-Level Static Attributes**: Trials (simulated surgical training runs) are established with static properties (e.g., patient ID, procedure type, operating room, base completion times, and overall performance metrics). An latent proficiency score ($p \in [0,1]$) is assigned using a Beta distribution:
   * Experts: $p \sim \text{Beta}(8, 2)$
   * Intermediates: $p \sim \text{Beta}(5, 5)$
   * Novices: $p \sim \text{Beta}(2, 8)$
2. **Kinematics & Tool Positions**: Tool trajectories are generated using continuous sine/cosine components simulating smooth sweeps. Amplitudes and high-frequency noise parameters are governed by the operator's skill level (Novices exhibit larger spatial movements and high jitter; Experts maintain focused, low-noise movement).
3. **Temporal Sequence Realism**: Every observation has a continuous simulated time element. Recording dates are generated as sequential datetime timestamps by mapping rows to sequential elapsed frames relative to a base date.
4. **Multivariate Correlations**: Features like completion times, errors, tissue collisions, and smoothness parameters are mathematically coupled to the latent proficiency score, establishing realistic dependencies without introducing perfectly linear relationships.
5. **Feature Embeddings**: Spatial (CNN) and temporal (Transformer) features represent deep embeddings extracted from hypothetical video encoders. They are modeled around distinct Gaussian cluster centers for the three skill levels to facilitate cluster analysis, dimensionality reduction, and classification.

---

## Statistical Correlation Assumptions
The dataset embodies several key surgical validation assumptions verified during generation:
* **Novices** show longer completion times, lower movement efficiency (high path length relative to straight path), high jerk index, more frequent tool pauses, high error counts, and lower precision.
* **Experts** show short completion times, highly efficient motion paths, high precision, low jerk, minimal collisions, and high dexterity.
* **Intermediates** represent a transitional performance profile.

### Computed Correlation Coefficients (Pearson $r$)
* **Skill Level vs. Completion Time**: -0.5893 (Strong negative correlation, indicating faster completion by experts)
* **Skill Level vs. Motion Smoothness**: 0.8758 (Strong positive correlation, indicating smoother motions by experts)
* **Skill Level vs. Precision Score**: 0.8675 (Strong positive correlation, indicating higher precision by experts)
* **Skill Level vs. Error Count**: -0.7680 (Strong negative correlation, indicating fewer errors by experts)

---

## Potential AI Applications
This dataset is suitable for training and evaluating several machine learning systems:
1. **Surgical Skill Classification**: Training classifiers (Random Forests, Gradient Boosting, SVMs, or Deep Neural Networks) to automatically grade surgeon performance from kinematic time series.
2. **Visual Embedding Analytics**: Utilizing the 5-dimensional CNN and Transformer features to perform unsupervised clustering (t-SNE, UMAP, PCA) representing video frame distributions.
3. **Event Detection & Safety Audits**: Regression models for predicting collision rates, or classifying safety risk categories based on tool distance and optical flow.
4. **Multi-Task Surgical Assessment**: Jointly predicting the success of tasks, overall performance scores, and error rates from time-window aggregates.

---

## Column Descriptions
For a full list of all 100 features, their descriptions, types, and ranges, please refer to the accompanying [data_dictionary.csv](file:///data_dictionary.csv).
