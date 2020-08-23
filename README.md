# Salary Prediction

Objective: Can we predict the right salary for the given job

## Table of contents
- Define Problem
    - Can we predict the right salary for the job description?
    - Goals

- Discover Data
   - Exploratory Data Analysis(EDA)
   - Data Visualization

- Develop solutions
    - Establish a baseline
    - Machine Learning Models
    - Hyperparameter Tuning & Optimization
    - Best model

- Deploy solution
   - Automate the pipeline
   - Save the predictions



## Defining The Problem

#### Can we predict the right salary for the job description?

If a company offers less salary then they will lose the candidate but if they offer him to much they will be wasting the resources. Is there a way to use AI to help us solve this complex problem for companies and candidates.

#### Goals

- Help companies secure the right candidate for the right salary
- Help candidates to secure the right position at the right salary

## Discovering the Data

#### Exploratory Data Analysis (EDA)
- we have 8 columns and 2 of them are id columns
Data columns (total 8 columns):

|#   |Column             |Non-Null Count    |Dtype|
|--- |------             |--------------    |----- |
| 0   |jobId              |  1000000 non-null|  object|
| 1   |companyId          |  1000000 non-null|  object|
| 2   |jobType            |  1000000 non-null|  object|
| 3   |degree             |  1000000 non-null|  object|
| 4   |major              |  1000000 non-null|  object|
| 5   |industry           | 1000000 non-null|  object|
| 6   |yearsExperience    |  1000000 non-null|  int64|
| 7   |milesFromMetropolis|  1000000 non-null|  int64|

- **year of experience** has a **positive correlation** of **0.375013** with salary
- **miles from metropolis** has a **negative correlation** of **-0.297686** with salary

|               |yearsExperience | milesFromMetropolis | salary
|-------------------| ------------------- | ------------------- |-------------------|
|yearsExperience |1.000000 |0.000672 |0.375013|
|milesFromMetropolis |0.000672 |1.000000 |-0.297686|
|salary|0.375013|-0.297686 |1.000000|

- **Ditribution table of years of experience and miles from metropolis**

|   |yearsExperience |milesFromMetropolis|
|----|--------------|-------------------|
|count |1000000.000000 |1000000.000000|
|mean |11.992386 |49.529260|
|std |7.212391 |28.877733|
|min |0.000000 |0.000000|
|25% |6.000000 |25.000000|
|50% |12.000000 |50.000000|
|75% |18.000000 |75.000000|
|max |24.000000 |99.000000|

#### Data Visualization

-- **Salary Distribution plot** --
<br />
-> We have more salaries around the range of 100 to 150 compared to 50-100 and 150-200

![Image of Distribution plot of salaries](/reports/figures/graphs/png/salary_distribution.png)

-- **job Type vs Salary box plot** --
<br />
-> It looks like the CTO and CEO gets paid more than everyone else and the janitor is lowest among all

![Image of job type vs salary box plot](/reports/figures/graphs/png/jobType_salary_box.png)

-- **Degree vs Salary violin plot** --
<br />
-> The Doctoral and Masters majors on average get paid more than the bachelors. It looks like the NONE adn Highs Schools get paid the least

![Image of Degree vs salary violin plot](/reports/figures/graphs/png/degree_salary_violin.png)

-- **Count plot of degree vs jobtype**--
<br />
My hypothesis is that to be a CFO of a company you need at least a masters maybe they forgot to mention the requirment of the degree

![Image of degree vs jobtype count plot](/reports/figures/graphs/png/jobType_count.png)
My hypothesis was wrong someone with high school diploma can become CEO or CFO

### Developing Solutions

#### Establishing the Baseline
- We used the mean salary of each industry to make our initial baseline predictions
- We got a **Mean Squared Error (MSE)** of **1367.1229507852554**

#### Machine Learning Models
- Hypothesized Models
- **XGBoost regressor** with hyperparameter tuning
- **Densely Neural Network(DNN)** with optimization and Hyperparameter tuning

#### Best Model
- Our Best model for now:
- With an MSE of **357.3861083984375**, a Densely connected neural network with two layers and Bayesian Optimization for hyperparameter tuning.

### Deploying Solution
- **Automated** the **pipeline** with one-hot encoding
- Predictions will be saved in an **EXCEL sheet** in a **comma Separated format(CSV)**

## Table of Contents
- Define Problem
    - Can we predict the right salary for the job description?
    - Goals

- Discover Data
   - Exploratory Data Analysis(EDA)
   - Data Visualization

- Develop solutions
    - Establish a baseline
    - Machine Learning Models
    - Hyperparameter Tuning & Optimization
    - Best model

- Deploy solution
   - Automate the pipeline
   - Save the predictions
