# Functional Consistency in Rehabilitation Movements

This repository accompanies a research project on **functional consistency of human movement across postural conditions** (standing, sitting, wheelchair) using 3D skeletal motion data.

The project focuses on **rehabilitation-relevant movement analysis**, integrating perspectives from **physiotherapy** and **occupational therapy**, and emphasizes **task-level motor strategies** rather than idealized biomechanical norms.

---

## 🔍 Project Rationale

In clinical rehabilitation, patients often perform the same functional task (e.g. upper-limb reaching) in different postural contexts due to physical, neurological, or environmental constraints.

Traditional biomechanical analyses typically evaluate movements against **standing normative patterns**, implicitly treating deviations as deficits.

This project adopts a different perspective:

> **A movement can be functionally valid even if it deviates from the standing pattern, as long as it achieves the task goal with a stable and repeatable strategy within the given context.**

We define and operationalize this concept as **functional consistency**.

---

## 🎯 Aim

To develop a **simple, interpretable methodology** that:

- Quantifies **task-level movement outcomes**
- Characterizes **movement strategies** within the same posture
- Assesses **functional consistency** across repeated executions
- Enables meaningful comparison across **standing**, **sitting**, and **wheelchair** conditions

---

## 🧠 Core Concept: Functional Consistency

A movement is considered **functionally consistent** if, within the same postural condition:

1. It achieves the intended **task goal** (e.g. sufficient hand elevation or reach)
2. It follows a **stable temporal pattern**
3. It employs a **repeatable coordination strategy** (e.g. trunk–arm relationship)

Importantly, **functional consistency is not judged by similarity to standing kinematics**, but by **within-condition repeatability and goal achievement**.

---

## 📊 Data Description

- Input data consist of **simplified skeletal motion files (.txt)**
- Each file represents **one complete movement execution**
- Each row corresponds to one frame
- Each row contains **75 values**:  
  25 joints × 3 coordinates (x, y, z)

### Postural Conditions
- `stand`
- `chair`
- `wheelchair`

### Gesture Labels
Gesture indices correspond to predefined upper- and lower-limb movements (e.g. elbow flexion, shoulder flexion, reaching), as defined in the source dataset.

---

## 🧪 Methodology Overview

For each movement clip:

1. **Root normalization**  
   - All joint positions are expressed relative to `SpineBase`

2. **Time normalization**  
   - Each movement is resampled to a fixed number of frames (e.g. 100)  
   - Enables fair comparison across executions of different durations

3. **Task-relevant joint selection**  
   - Upper-limb gestures → shoulder–elbow–wrist–hand chain  
   - Lower-limb gestures → hip–knee–ankle–foot chain

4. **Feature extraction (task-level)**  
   - Joint range of motion (ROM)
   - End-point displacement and path length
   - Peak and mean speed
   - Movement smoothness (normalized jerk)
   - Stability / variability indicators

5. **Within-condition comparison**  
   - Movements are compared **only within the same posture**
   - Consistency is evaluated based on similarity of outcomes and strategies

---

## 🩺 Clinical Interpretation

The extracted features are intentionally chosen to support **clinical reasoning**:

- **Physiotherapy**  
  - Movement amplitude  
  - Speed and smoothness  
  - Strength and control demands  

- **Occupational Therapy**  
  - Task achievement  
  - Strategy selection  
  - Adaptation to context and constraints  

This framework supports a **capability-oriented view of movement**, focusing on *how the body solves a task*, rather than how closely it matches an idealized pattern.

---

## 🛠 Repository Structure

```text
.
├── data/                 # Raw or example motion files
├── analysis/             # Feature extraction and comparison scripts
├── figures/              # Generated plots and visualizations
├── README.md             # Project description (this file)
└── requirements.txt      # Python dependencies
