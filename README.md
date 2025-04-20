# XCare AI - Personal Skincare Consultant

XCare AI is an intelligent virtual assistant built on advanced generative AI technologies. The bot provides personalized skincare recommendations, cosmetic product selection, and care routines.

## Main Features

- Personalized cosmetic product selection
- Scientifically-backed recommendations
- Skin condition analysis through photos
- Detailed step-by-step care instructions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/universescreatorsarm/xcare-ai.git
cd xcare-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Running the Application

1. Start the server:
```bash
uvicorn app:app --reload
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

## Project Structure

```
xcare-ai/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── .env               # API keys file
├── data/              # Cosmetic products dataset
├── static/            # Static files (logo)
└── templates/         # HTML templates
```

## Usage

1. Enter your question in the text field
2. Upload a photo for skin condition analysis
3. Receive personalized recommendations

## Technologies

- FastAPI - web framework
- Google Gemini - generative AI model
- LangChain - LLM framework
- FAISS - vector storage for RAG
- Pandas - data processing 
