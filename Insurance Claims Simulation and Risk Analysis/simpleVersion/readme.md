# Insurance Claims Simulation — Beginner Project
## Overview

This project simulates one year of insurance claims to explore how data can be generated, analyzed, and visualized in Python.
It is designed for beginners who want to understand basic data simulation and plotting without complex machine learning.

## Objectives

Simulate random daily insurance claims for a whole year

Categorize claims into Car, Home, or Health

Analyze average claim sizes and total claim counts

Visualize trends and distributions

🛠️ Tools & Libraries

Python – Core programming language

Pandas – For creating and manipulating data tables

NumPy – For generating random numbers and calculations

Matplotlib – For plotting and visualizing results

## How It Works

Create a Date Range

Generate every day from Jan 1, 2024 to Dec 31, 2024

Simulate Claims

Number of claims per day is random using a Poisson distribution (average = 5/day)

Claim amounts follow an Exponential distribution (average = $2000)

Each claim is randomly assigned a category: Car, Home, or Health

Build a DataFrame

Store all simulated claims in a Pandas table with date, size, and category

Summarize Results

Calculate average claim size

Count total claims

Visualize

Plot claims per day over time

Plot distribution of claim sizes

## Example Output
#### Daily Claims Count

Shows how many claims occur each day throughout the year.


#### Claim Size Distribution

Shows how claim sizes are spread (many small claims, few big claims).
