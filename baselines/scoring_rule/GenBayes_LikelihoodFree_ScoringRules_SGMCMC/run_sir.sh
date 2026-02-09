#!/bin/bash
set -e

source .venv/bin/activate
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