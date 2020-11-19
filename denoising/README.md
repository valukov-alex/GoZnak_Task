# Denoising на mel-спектрограммах
Задача: избавить mel-спектрограмму от шума  
## Обучение:  
```
python3 train.py train_dir val_dir
```
В папках train_dir и val_dir должны лежать папки clean и noisy  
Так же есть опциональные аргументы:  
-m save_model_path - путь для сохранения модели, по умолчанию - ./models/model.pt  
-e epochs - кол-во эпох, по умолчанию 100   
-l learning_rate, по умолчанию 1e-5  
## Запуск:  
```
python3 run.py path_to_file path_to_save
```
Опциональный аргумент:  
-m model_path - путь к модели, по умолчанию ./models/model.pt  
Скрипт берет пектрограмму из path_to_file и сохраняет результат в path_to_save