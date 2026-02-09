source .venv/bin/activate &&
python exp.py sir_undercounting false false false 100 && 
python exp.py sir_undercounting true false false 1 &&
python exp.py sir_undercounting true false false 10 &&
python exp.py sir_undercounting true false false 1000 &&
python exp.py sir_undercounting true false false 10000 &&
python exp.py sir_undercounting true true true 100 && # pick best beta
python exp.py sir_student_t_1 false false false 100 && 
python exp.py sir_student_t_1 true false false 1 &&
python exp.py sir_student_t_1 true false false 10 &&
python exp.py sir_student_t_1 true false false 1000 &&
python exp.py sir_student_t_1 true false false 10000 &&
python exp.py sir_student_t_1 true true true 100 &&
python exp.py sir_student_t_2 false false false 100 && 
python exp.py sir_student_t_2 true false false 1 &&
python exp.py sir_student_t_2 true false false 10 &&
python exp.py sir_student_t_2 true false false 1000 &&
python exp.py sir_student_t_2 true false false 10000 &&
python exp.py sir_student_t_2 true true true 100
