python -m venv venv
source venv/bin/activate       # mac/linux
# venv\Scripts\activate        # windows

pip install -r requirements.txt

python translate.py

python translate.py -t "I love programming in Python."

# make sentences.txt with one english sentence per line
python translate.py -f sentences.txt -b 16

uvicorn api:app --reload --host 0.0.0.0 --port 8000

curl -X POST "http://127.0.0.1:8000/translate" -H "Content-Type: application/json" -d '{"text":"I love open source software."}'