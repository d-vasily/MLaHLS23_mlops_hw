# Проект по млопс

В качестве задачи выбрана работа с временным рядом: предсказание цены закрытия акций.



For usage:

1. install miniconda (or `virtualenv` although not tested)
2. create env
```
conda create -n test_env
```
3. activate env
```
conda activate test_env
```
4. install python
```
conda install python=3.10
```
5. install poetry
```
conda install poetry
```
6. install necessary packages
```
poetry install
```
7. install necessary programms
```
pre-commit install
```
8. hook checking
```
pre-commit run -a
```
9. run train.py (create folder models/ and save model to file)
```
python train.py
```
10. run infer.py (save prediction to data/predictions/)
```
python infer.py
```
