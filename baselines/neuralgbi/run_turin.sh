source .venv/bin/activate &&
# python exp.py turin false false false 100 && 
python exp.py turin true false false 1 &&
python exp.py turin true false false 10 &&
python exp.py turin true false false 1000 &&
python exp.py turin true false false 10000 &&
python exp.py turin true true true 100