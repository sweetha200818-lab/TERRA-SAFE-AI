@echo off
py -m pip install -r requirements.txt
py -m uvicorn app:app --reload
pause
