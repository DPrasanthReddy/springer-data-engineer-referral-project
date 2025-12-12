# springer-data-engineer-referral-project
Take-Home Test for Data Engineer Intern

This project implements a complete data profiling, transformation, fraud-detection logic, and reporting pipeline for Springer Capital’s user referral program.
The solution is built using Python (Pandas) and packaged with Docker, following all instructions provided in the assignment PDF.

Project Overview

The objective of this assignment is to:

Load and clean referral-related data from multiple sources

Profile all tables

Join the data and apply business logic to identify potential fraud

Generate a final report of 46 records

Containerize the solution using Docker

Provide documentation and a data dictionary

This repository contains all required deliverables.

Repository Structure
springer-referral-project
│
├── main.py                       # Main data processing pipeline
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker setup for container execution
├── README.md                     # Project documentation (this file)
│
├── data/                         # (Uploaded by user)
│   ├── user_referrals.csv
│   ├── user_referral_logs.csv
│   ├── user_referral_statuses.csv
│   ├── user_logs.csv
│   ├── referral_rewards.csv
│   ├── paid_transactions.csv
│   ├── lead_log.csv
│
└── output/                       # Pipeline results
    ├── final_output.csv          # Final 46-row processed report
    ├── profiling_report.csv      # Data profiling result
    ├── data_dictionary.xlsx      # Business-readable data dictionary

How the Pipeline Works

The core logic is inside main.py, which:

1. Loads all CSV files

Every source table is read into a pandas DataFrame.

2. Profiles all tables

For each column, the script computes:

number of rows

null count

distinct value count
The full profiling output is stored as:

output/profiling_report.csv

3. Cleans and standardizes data

Datetime columns parsed

String fields Title-cased (InitCap)

Clubs remain unmodified

Reward values automatically parsed

Missing values treated safely

4. Timezone handling

All timestamps are stored in UTC.
The script converts them to local time using:

user's homeclub timezone

transaction timezone

lead timezone

default fallback: Asia/Jakarta

5. Joins all source tables

Merges all datasets required to compute referral integrity:

user_referrals

user_referral_logs

user_logs

user_referral_statuses

referral_rewards

paid_transactions

lead_log

6. Computes referral_source_category

Using logic:

User Sign Up     → Online
Draft Transaction → Offline
Lead              → leads.source_category

7. Applies full business logic to detect fraud

Each referral is evaluated against all conditions provided in the assignment PDF:

Valid Condition 1

Valid Condition 2

Invalid Conditions 1–5

Final columns:

is_business_logic_valid (TRUE / FALSE)

business_logic_reason (rule explanation)

8. Generates final report (46 rows)

According to assignment requirements:

Include all valid referrals

Add pending or failed no-reward referrals

Fill remaining slots using the latest referrals

Result saved as:

output/final_output.csv

Running the Project (Local Machine)
Install dependencies
pip install -r requirements.txt

Run the pipeline
python main.py --input data --output output

Running with Docker
Build the image
docker build -t springer-referral .

Run the container
docker run -v "$(pwd)/output:/app/output" springer-referral


This ensures that final_output.csv, profiling file, and dictionary are saved outside the container.

Deliverables Included
Deliverable	File
Python script	main.py
Data Profiling	output/profiling_report.csv
Final 46-row Report	output/final_output.csv
Documentation (README)	README.md
Data Dictionary	output/data_dictionary.xlsx
Dockerfile	Dockerfile
Requirements file	requirements.txt

All required files for the assignment have been included exactly as requested.
 Data Dictionary (Summary)

A full Excel version is available in:

output/data_dictionary.xlsx


It includes column name, description, datatype, and notes for business users.

