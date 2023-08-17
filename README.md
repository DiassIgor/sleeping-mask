# Sleeping Mask Project

This project will be divided in two parts:
  1. Oura Ring Analysis
  2. Sleeping Mask Validation

## 1. Oura Ring Analysis

The Oura Ring data analysis is made up:
  - Data Extraction: Collect the data from the Oura API, and extract the bpm, time and hypnogram information
  - Exploratory Data Analysis: Analyse and understand the variables, as well as identify the limitations of the data
  - Model Training: Try to apply ML and DL models in the heart rate data with the purpose of creating the equivalent hypnogram
  - Web Development: Try to replicate the Cloud Oura API dashboards   

## 2. Sleeping Mask Validation

The usage of the EOG signals for predicting the REM during the sleep can be validated with the Oura Ring Data. In addition, its possible combine another features extracted by the Oura Ring to improve EOG analysis. 

Above the set of measures in Oura Ring that is possible useful:
  - Bedtime Start
  - Bedtime End
  - Awake Sleep Time
  - REM Sleep Time
  - Light (N1 or N2) Sleep Time
  - Deep (N3) Sleep Time
  - Heart Rate Measure (Lowest, Highest and Average)
  - Breath Average
  - Hypnogram

The Hypnogram Informartion can be useful for REM stage comparison between the Sleep Mask data and the Oura Ring Data

In addition, the Oura Ring Analysis phase can be useful for generate insights about data storage, data preprocessing and model building

The replicated API in the web development phase will be used as a template of sleeping mask API
