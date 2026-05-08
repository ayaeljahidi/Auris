.venv\Scripts\activate  
pip install modelscope[audio] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
python scripts/setup_models.py 
uvicorn backend.main:app --reload --port 8000 --reload-dir backend