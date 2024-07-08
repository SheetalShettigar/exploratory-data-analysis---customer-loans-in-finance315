# Project Title: Exploratory Data Analysis - Customer Loans in Finance

To perform exploratory data analysis on the loan portfolio, using various statistical and data visualisation techniques to uncover patterns, relationships, and anomalies in the loan data.

## Table of Contents:
1. Project Description
1. Installation instructions
1. Usage instructions
1. File structure of the project
   
## Project Description

Is to perform exploratory data analysis on the loan portfolio, using various statistical and data visualisation techniques to uncover patterns, relationships, and 
anomalies in the loan data. 

This information will enable the business to make more informed decisions about loan approvals, pricing, and risk management.
The aim is to gain a deeper understanding of the risk and return associated with the business' loans by conducting exploratory data analysis on the loan data

Ultimate goal is to improve the performance and profitability of the loan portfolio,

## Python Files
###db_utils.py
###data_transform.py
## Usage instructions
Run db_utils.py to establish connection with the database. To read the data from loan_payments table into  a csv file.
Load the data into the panadas dataframe for analyisis.

For EDA analysis of the loan payments file run the data_transform.py file 

## File structure of the project
###db_utils.py
class RDSDatabaseConnector
methods:
get_connection() returns engine connection
read_data(engine_connection) returns dataframe
data_to_csv(data) 

###data_transform.py
Class DataTransform
methods:
check_data()
transform_data()
check_outliers()
check_skew()
check_colinearity()
percentage_recovered()
loss_to_company()
loss_indicators()



