# Проект по млопс

В качестве задачи выбрана работа с временным рядом: предсказание цены закрытия акций.



For usage:

1. install miniconda (or `virtualenv` although not tested)
2. conda create -n test_env (create env)
3. conda activate test_env (activate env)
4. conda install python=3.10 (install python)
5. poetry install (install necessary packages)
6. pre-commit install 
7. pre-commit run -a (hook checking)
8. python train.py (create folder models/ and save model to file)
10. python infer.py (save prediction to data/predictions/)
