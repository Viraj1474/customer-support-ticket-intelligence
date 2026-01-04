# Smart System for Handling Customer Support Tickets

Every day, customer support teams receive a large number of requests. Manually reviewing each ticket, identifying urgent cases, and noticing frustrated customers takes time and effort. Without better tools, response quality and efficiency can drop quickly.

This project focuses on building a support operations dashboard that helps teams:

1. See which tickets are currently active  
2. Identify tickets that need faster responses  
3. Highlight customers who may be dissatisfied or at risk of leaving  

The system is designed to support decision-making, not to replace human judgment.

---

## Problem Statement

In real customer support environments:

- Ticket descriptions are often short and repetitive  
- Ticket priority is assigned based on operational context, not just text content  
- Customer churn depends on multiple factors, not a single event  

Response speed matters, but customer satisfaction and repeated unresolved issues play a larger role. A single delayed or poor interaction may not cause churn, but consistent negative patterns often do.

Because of this, relying only on high-accuracy machine learning models is rarely practical. This project takes a different approach by combining basic NLP, structured data, and clear business rules to generate useful insights.

---

## What This Project Does

1. Performs basic text preprocessing on ticket subject and description  
2. Builds baseline models for ticket type and ticket priority as exploratory steps  
3. Identifies customers likely to churn using understandable patterns  
4. Displays insights through a clean Streamlit dashboard  
5. Allows filtering and exporting of high-risk customer data  

---

## Dataset Description

The dataset contains customer support records with information such as:

1. Ticket subject and description  
2. Ticket priority and status  
3. Customer satisfaction rating  
4. Ticket channel  
5. Purchase date and resolution time  
6. Customer identifier to detect repeated issues  

Messages often follow a fixed pattern, which reflects how real support tickets are written in practice.

---

## Approach Used

### Text Processing
1. Ticket subject and description are combined  
2. Text is cleaned using simple NLP techniques  
3. TF-IDF is used for vectorization  

### Machine Learning
1. Logistic Regression is used as a baseline model  
2. The model helps explore patterns in the data  
3. Accuracy is limited due to similar wording across tickets  

These models are treated as experiments, not final decision-makers.

---

## Churn Risk Logic

Churn risk is calculated using transparent and interpretable rules:

1. Low customer satisfaction  
2. High or critical ticket priority  
3. Long resolution times  
4. Multiple tickets from the same customer  

Each factor contributes to a single churn score, which is classified as:

- Low risk  
- Medium risk  
- High risk  

This rule-based approach is easy to explain and aligns with how support teams work in practice.

---

## Dashboard

The Streamlit dashboard provides:

1. A view of currently active tickets  
2. Filters based on ticket priority and churn risk  
3. A list of customers requiring immediate follow-up  
4. Export functionality for further analysis  

The dashboard recalculates churn logic directly from the source data, ensuring consistent results.

---

## Project Structure

customer-support-ticket-intelligence/
├── app.py
├── README.md
├── requirements.txt
├── notebooks/
│ └── exploratory notebooks
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── churn_analysis.py
│ └── models.py
└── data/
└── raw/


---

## Assumptions and Limitations

1. Ticket descriptions are repetitive, limiting text-based classification accuracy  
2. Purchase date is used as a proxy when ticket creation time is unavailable  
3. Churn risk is rule-based, not predictive  
4. The system is intended for analysis and decision support, not live deployment  

These limitations reflect real-world data challenges.

---

## How to Run the Project

1. Clone the repository  
2. Create and activate a virtual environment  
3. Install dependencies:  
pip install -r requirements.txt
4. Run the dashboard:  
streamlit run app.py


---

This project focuses on practical decision support rather than artificial accuracy. It demonstrates how data analysis and simple machine learning can assist customer support teams in real scenarios.
