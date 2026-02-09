#!/bin/bash
set -e

source .venv/bin/activate

python experiments/exp.py gnk --index 0 && sleep 5
python experiments/exp.py gnk --index 1 && sleep 5
python experiments/exp.py gnk --index 2 && sleep 5
python experiments/exp.py gnk --index 3 && sleep 5
python experiments/exp.py gnk --index 4 && sleep 5
python experiments/exp.py gnk --index 5 && sleep 5
python experiments/exp.py gnk --index 6 && sleep 5
python experiments/exp.py gnk --index 7 && sleep 5
python experiments/exp.py gnk --index 8 && sleep 5
python experiments/exp.py gnk --index 9 && sleep 5
python experiments/exp.py gnk --index 10 && sleep 5
python experiments/exp.py gnk --index 11 && sleep 5
python experiments/exp.py gnk --index 12 && sleep 5
python experiments/exp.py gnk --index 13 && sleep 5
python experiments/exp.py gnk --index 14 && sleep 5
python experiments/exp.py gnk --index 15 && sleep 5
python experiments/exp.py gnk --index 16 && sleep 5
python experiments/exp.py gnk --index 17 && sleep 5
python experiments/exp.py gnk --index 18 && sleep 5
python experiments/exp.py gnk --index 19 && sleep 5
python experiments/exp.py sir_undercounting --index 0 && sleep 5
python experiments/exp.py sir_undercounting --index 1 && sleep 5
python experiments/exp.py sir_undercounting --index 2 && sleep 5
python experiments/exp.py sir_undercounting --index 3 && sleep 5
python experiments/exp.py sir_undercounting --index 4 && sleep 5
python experiments/exp.py sir_student_t_1 --index 0 && sleep 5
python experiments/exp.py sir_student_t_1 --index 1 && sleep 5
python experiments/exp.py sir_student_t_1 --index 2 && sleep 5
python experiments/exp.py sir_student_t_1 --index 3 && sleep 5
python experiments/exp.py sir_student_t_1 --index 4 && sleep 5
python experiments/exp.py sir_student_t_2 --index 0 && sleep 5
python experiments/exp.py sir_student_t_2 --index 1 && sleep 5
python experiments/exp.py sir_student_t_2 --index 2 && sleep 5
python experiments/exp.py sir_student_t_2 --index 3 && sleep 5
python experiments/exp.py sir_student_t_2 --index 4 && sleep 5