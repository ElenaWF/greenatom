# Тестовое задание на стажировку "Гринатом"
![Метрики](https://raw.githubusercontent.com/ElenaWF/greenatom/master/3.png)

Данные: открытый набор данных, который содержит в себе отзывы о фильмах, а также соответствующие им оценки рейтинга.

Задача:
1. Обучить модель на языке Python для классификации отзывов.
2. Разработать веб-сервис на базе фреймворка Django для ввода отзыва о фильме с автоматическим присвоением рейтинга (от 1 до 10) и статуса комментария (положительный или отрицательный).

Ссылка на веб-сервис: http://supermi3.beget.tech/

Для запуска приложения на Windows локально в папке выполните команды
1. py -m venv env
2. .\env\Scripts\activate
3. pip install -r requirements.txt
4. py manage.py runserver

Проект будет доступен локально http://127.0.0.1:8000/ .
В папке files - предобученные модели.
