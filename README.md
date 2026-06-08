# Robustness Evaluation of IoT Intrusion Detection Systems

## Abstract

This repository contains the implementation and experimental framework for evaluating the adversarial robustness of Machine Learning-based Intrusion Detection Systems (IDS) in Internet of Things (IoT) environments.

The study investigates the vulnerability of IDS models to adversarial examples generated using Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). In addition, several defense mechanisms are evaluated to analyze their effectiveness in improving model robustness against adversarial attacks.

## Research Objectives

The objectives of this research are:

* To develop a machine learning-based IDS for IoT environments.
* To evaluate the robustness of IDS models under adversarial attacks.
* To compare the performance of Random Forest, XGBoost, and Multi-Layer Perceptron (MLP).
* To analyze the impact of different perturbation levels on detection performance.
* To investigate the effectiveness of defense mechanisms such as Adversarial Training and Feature Squeezing.

## Dataset

Experiments are conducted using the N-BaIoT dataset, a publicly available benchmark dataset for IoT botnet detection. The dataset contains benign and malicious network traffic collected from multiple IoT devices under realistic attack scenarios.

## Models

The following machine learning models are evaluated:

* Random Forest
* XGBoost
* Multi-Layer Perceptron (MLP)

## Adversarial Attacks

The robustness evaluation is performed using:

* Fast Gradient Sign Method (FGSM)
* Projected Gradient Descent (PGD)

Perturbation levels:

ε ∈ {0.1, 0.3, 0.5}

## Defense Mechanisms

The following defense techniques are implemented:

* Adversarial Training
* Feature Squeezing
* Gaussian Smoothing

## Evaluation Metrics

Performance is assessed using:

* Accuracy
* Precision
* Recall
* F1-Score
* Attack Success Rate (ASR)
* False Positive Rate (FPR)
* False Negative Rate (FNR)
* Robust Accuracy
* Robustness Score

## Project Structure

```text
data/
models/
attacks/
defenses/
evaluation/
results/
notebooks/
README.md
```

## Research Contributions

This project provides:

1. A reproducible robustness evaluation pipeline for IoT IDS.
2. Comparative analysis of Random Forest, XGBoost, and MLP under adversarial conditions.
3. Quantitative evaluation of FGSM and PGD attacks on tabular IoT network traffic.
4. Assessment of defense mechanisms for improving adversarial robustness.
5. Analysis of the trade-off between clean performance and robustness.

## Author

Nguyen Thi Huyen Tram

Information Security Program

University of Economics Ho Chi Minh City (UEH)

Research Interests: Cybersecurity, IoT Security, Intrusion Detection Systems, Adversarial Machine Learning
