@echo off
REM Create virtual environment if it does not exist
if not exist "env\Scripts\python.exe" (
    echo Creating Python virtual environment...
    python -m venv env
)
REM Activate virtual environment
call env\Scripts\activate
REM Install requirements
pip install -r requirements.txt
REM Run the main Python file
python main.py