# Facebook Marketing API to Database Pipeline

A Python ETL pipeline that extracts campaign data from Facebook's Marketing API, transforms it, and loads it into a PostgreSQL database.

## 📌 Features

- Automated daily extraction of campaign insights & status
- Data validation and processing with Pandas
- Secure credential management using environment variables
- Error handling and logging at each pipeline stage
- Timezone-aware date handling (Asia/Hong_Kong)

## Project Structure

facebook_api/
├── facebook_api.py  
├── requirements.txt  
├── .env  
├── .gitignore  
└── README.md

## ⚙️ Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database connected to AWS data warehouse
- Facebook Marketing API access

### Installation

1. Clone the repository
2. Make sure Facebook developer credentials and Database credentials are saved securely as environment variables (.env or AWS secrets manager)
3. python -m venv venv
   source venv/bin/activate # Linux/Mac
   .\venv\Scripts\activate # Windows
4. pip install -r requirements.tx

### Execution

1. in terminal: `python facebook_api.py`
