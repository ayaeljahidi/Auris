.venv\Scripts\activate  
python scripts/setup_models.py 
uvicorn backend.main:app --reload --port 8000 --reload-dir backen