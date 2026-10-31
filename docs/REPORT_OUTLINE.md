# CENG391 Term Project Report: Section and Subsection Outline

Main sections are the large headings between the horizontal lines. Subsections are listed under each one, with a note on what to write.

---

# ABSTRACT

---

*Single paragraph, at most about 250 words. Written last.*

---

# SECTION 1: INTRODUCTION

---

**1.1 Motivation and Application Context**

*Why touchless gesture control is useful, with two or three concrete use cases.*

**1.2 Problem Statement**

*What the system must infer from camera images and why this is hard: lighting, background, distance, user differences, intentional versus incidental movement.*

**1.3 Aim and Objectives**

*One-sentence aim and a list of objectives.*

**1.4 Proposed Approach**

*The pipeline in short, from camera input to device command.*

**1.5 Contributions**

*What this project delivers, in three or four points.*

**1.6 Report Organization**

*One sentence per remaining section.*

---

# SECTION 2: RELATED WORK

---

**2.1 Hand Detection and Landmark Estimation**

*Methods for finding the hand and its keypoints, and the one we use.*

**2.2 Gesture Datasets and Benchmarks**

*Existing datasets, their size, classes, static or dynamic content, and fit for device control.*

**2.3 Static and Dynamic Gesture Recognition**

*Landmark-based and video-based methods, classical and learned classifiers.*

**2.4 Continuous Recognition and False-Trigger Prevention**

*How existing work finds gesture start and end, rejects non-gesture motion, and measures false positives and latency.*

**2.5 Temporal Smoothing**

*How noisy landmark streams are filtered and the jitter and lag trade-off.*

**2.6 Comparison and Positioning**

*Table of cited works and a short statement of how this project differs.*

---

# SECTION 3: DATASET AND PROBLEM DEFINITION

---

**3.1 Task Definition**

*The exact computer vision tasks: static classification, dynamic classification, segmentation.*

**3.2 Gesture Set and Command Mapping**

*Table of every gesture, its type, and the command it triggers.*

**3.3 Data Collection Protocol**

*Camera, recording procedure, instructions given to participants.*

**3.4 Participants and Recording Conditions**

*Number of users, backgrounds, lighting conditions, camera distances.*

**3.5 Dataset Statistics**

*Samples per class, per user, and per condition.*

**3.6 Train, Validation, and Test Split**

*Person-based split and how it is organized.*

**3.7 Preprocessing and Augmentation**

*Normalization steps and augmentations used.*

**3.8 Data Limitations**

*Known biases and gaps in the data.*

---

# SECTION 4: METHODOLOGY

---

**4.1 System Overview**

*Pipeline diagram and a short description of each stage.*

**4.2 Hand Detection and Landmark Extraction**

*Model, settings, and the output used.*

**4.3 Landmark Normalization and Features**

*Formulas for position, scale, and any derived features.*

**4.4 Temporal Smoothing**

*Filter, parameters, and how they were chosen.*

**4.5 Gesture Segmentation**

*How start and end of a gesture are detected, thresholds, and hysteresis.*

**4.6 Static Gesture Classification**

*Models, inputs, training, and hyperparameters.*

**4.7 Dynamic Gesture Classification**

*Window length, models, training, and hyperparameters.*

**4.8 User Calibration**

*What is measured during calibration and which parameters it adjusts.*

**4.9 False-Trigger Prevention**

*Confidence threshold, hold time, activation gesture, cooldown, and the no-gesture class.*

**4.10 Command Mapping and Device Control**

*How recognized gestures become commands.*

**4.11 Implementation Details**

*Language, libraries and versions, hardware, and how to run the code.*

---

# SECTION 5: EXPERIMENTS AND RESULTS

---

**5.1 Experimental Setup**

*Hardware, software, test procedure, and participants.*

**5.2 Evaluation Metrics**

*Definition of every metric used.*

**5.3 Baseline versus Learned Classifier**

*Experiment E1.*

**5.4 Effect of Temporal Smoothing**

*Experiment E2.*

**5.5 Effect of Calibration**

*Experiment E3.*

**5.6 Effect of False-Trigger Prevention**

*Experiment E4.*

**5.7 Cross-User Evaluation**

*Experiment E5.*

**5.8 Effect of Background**

*Experiment E6.*

**5.9 Effect of Lighting**

*Experiment E7.*

**5.10 Effect of Camera Distance**

*Experiment E8.*

**5.11 Runtime Performance**

*Experiment E9: FPS and end-to-end latency.*

**5.12 Qualitative Results**

*Example frames and live demonstration screenshots.*

---

# SECTION 6: DISCUSSION

---

**6.1 Interpretation of Results**

*Why the system performs as observed.*

**6.2 Success and Failure Cases**

*Which gestures and conditions work or fail, with examples.*

**6.3 Limitations**

*Technical limits of the system.*

**6.4 Computational Considerations**

*Speed and resource use.*

**6.5 Threats to Validity**

*Number of users, camera, environment, and how far results generalize.*

---

# SECTION 7: CONCLUSION AND FUTURE WORK

---

**7.1 Conclusion**

*Problem, approach, key findings, and whether the objectives were met.*

**7.2 Future Work**

*Realistic improvements and extensions.*

---

# SECTION 8: TEAM CONTRIBUTIONS

---

*One entry per group member: literature review, data preparation, implementation, experiments, evaluation, report, presentation.*

---

# REFERENCES

---

*Generated from literature.bib.*

