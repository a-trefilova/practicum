import pandas as pd
import os
import numpy as np
import nltk
from keras.src.optimizers import Adam
from nltk.corpus import stopwords

import time
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

pd.set_option('display.max_columns', None)

# Выставляем персональные данные для подключения к API Kaggle
kaggle_username = "alenasabitskaya"
kaggle_key = "e45f814cd9075e574865c3b0b8e300f7"

# Записываем данные в переменные окружения
os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

# Метод для получения данных
def read_data_from_csv_file():
    import tempfile
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Инициализируем объект KaggleApi для чтения данных
    api = KaggleApi()

    # Выполняем аутентификацию - на этом шаге метод сам возьмет user_name и ключ из переменных окружения
    api.authenticate()

    # Инициализируем необходимые константы для доступа к файлу

    # 1) Имя владельца датасета
    dataset_owner = 'snehaanbhawal'
    # 2) Имя датасета
    dataset_name = 'resume-dataset'
    # 3) Путь до файла с датасетом
    dataset_file_name = 'Resume/Resume.csv'

    # Используем временную директорию для сохранения и чтения датасета
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1) Определяем путь, по которому будет располагаться будущий сохраненый датасет
        path = temp_dir

        # 2) Определяем, требуется ли распаковка из zip формата
        is_unzip_needed = True

        # 3) Скачиваем данные, используя метод API dataset_download_files
        api.dataset_download_files(f'{dataset_owner}/{dataset_name}', path=path, unzip=is_unzip_needed)

        # 4) Формируем полный путь до csv файла
        csv_file_path = os.path.join(path, dataset_file_name)

        # 5) Читаем csv файл из полного пути
        data = pd.read_csv(csv_file_path)

        # 6) Возвращаем данные из файла
        return data

# Используем метод получения данных
data = read_data_from_csv_file()

# Метод для очистки данных
def clean_data(data):

    # Получаем количество строк в неочищенных данных
    rows_count = len(data)

    # Печатаем количество строк в неочищенных данных
    print(rows_count)

    # Выполняем удаление всех строк, где хотя бы в одной ячейке отсутствуют данные
    cleaned_data = data.dropna()

    # Получаем количество строк в очищенных данных
    cleaned_rows_count = len(cleaned_data)

    # Печатаем количество строк в очищенных данных
    print(cleaned_rows_count)

    # Определяем, требуется ли очистка данных:
    # Если строк с очищенными данными меньше,
    # значит очистка была эффективна, и тогда
    # возвращаем очищенные данные, иначе
    # очистка не требуется, и возвращаем исходные данные
    if cleaned_rows_count < rows_count:
        return cleaned_data
    else:
        return data

# Используем метод очистки данных
cleaned_data = clean_data(data)

# Создаем дополнительный массив стоп-слов
additional_stop_words = [
    'name', 'city', 'state', 'country',
    'fullname', 'company', 'resume', 'address',
    'phone', 'email', 'e-mail', 'summary',
    'experience', 'education', 'skill', 'skills',
    'contact', 'detail', 'details', 'mail',
    'website', 'web', 'url', 'www', 'year',
    'month', 'day', 'hour', 'hours', 'practice'
]

# Скачиваем локально стоп-слова
nltk.download('stopwords')
nltk.download('punkt')

# Формируем финальный список слов
stop_words = set(stopwords.words('english')+additional_stop_words)

# Метод очистки текста от лишних символов
def text_cleaning(raw_text):
    import re as regex
    import string
    from nltk.tokenize import word_tokenize

    # Считаем количество символов в строке до всей очистки
    total_symbols_count = len(raw_text)

    # Убираем все ссылки из резюме, которые начинаются с http или https
    raw_text_without_urls = regex.sub('http[s]?\\S+\\s*', ' ', raw_text)

    # Используем все знаки пунктуации, которые есть в string.punctuation
    translator = str.maketrans('', '', string.punctuation)

    # Очищаем строку от всех знаков пунктуации
    cleaned_text = raw_text_without_urls.translate(translator)

    # Определяем список названий месяцев и их сокращений в английском языке
    months = ["January", "Jan", "February", "Feb", "March", "Mar", "April", "Apr",
              "May", "June", "Jun", "July", "Jul", "August", "Aug",
              "September", "Sep", "October", "Oct", "November", "Nov", "December", "Dec"]

    # Пишем регулярное выражение для удаления месяцев
    months_regex = r'\b(' + '|'.join(months) + r')\b'

    # Применяем регулярное выражение к нашему
    # тексту, независимо от регистра
    cleaned_text = regex.sub(months_regex, '', cleaned_text, flags=regex.IGNORECASE)

    # Пишем регулярное выражение для удаления всех цифр
    digits_regex = r'\d+'

    # Применяем регулярное выражение удаления всех цифр
    cleaned_text = regex.sub(digits_regex, '', cleaned_text)

    # Токенизируем текст
    tokens = word_tokenize(cleaned_text)

    # Удаляем все стоп-слова согласно токенам
    filtered_words = [word for word in tokens if word.lower() not in stop_words]

    # Склеиваем слова обратно в строку
    cleaned_text = ' '.join(filtered_words)

    # Заменяем все множественные пробелы одним
    cleaned_text = ' '.join(cleaned_text.split())

    # Считаем количество символов в строке после всей очистки
    cleaned_symbols_count = len(cleaned_text)

    # Возвращаем очищенную строку со строчными символами и количество удаленных символов
    return cleaned_text.lower(), total_symbols_count - cleaned_symbols_count

# Применяем метод очистки текста от
# лишних символов к каждой ячейке
# в колонке Resume_str и записываем эти
# данные в две другие колонки
cleaned_data[['Cleaned_Resume_str', 'Removed_Symbols']] = cleaned_data['Resume_str'].apply(
    lambda x: pd.Series(text_cleaning(x))
)

# Печатаем в консоль 10 записей из таблицы
print(cleaned_data.sample(10))

# Считаем общее количество удаленных символов в ходе очистки
total_removed_symbols = cleaned_data['Removed_Symbols'].sum()

# Печатаем в консоль общее количество удаленных символов
print(total_removed_symbols)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Объявляем время начала обучения модели
start_time = time.time()

# Объявляем переменную, в которой содержатся все резюме
all_resume_values = cleaned_data['Cleaned_Resume_str'].values

# Объявляем таргет переменную, в которой содержатся категории
all_categories = cleaned_data['Category'].values

# Инициализируем объект класса,
# который отвечает за векторизацию текста
tf_idf_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')

# Используем объект для представления текста в виде векторов
all_resume_values_vectors = tf_idf_vectorizer.fit_transform(all_resume_values)

# Инициализируем объект класса,
# который отвечает за нормализацию
# целевых данных
label_encoder = LabelEncoder()

# Применяем метод нормализации
# к целевым данным
encoded_categories = label_encoder.fit_transform(all_categories)

# Преобразуем нормализованные числовые данные
# в матрицу
encoded_categories_matrix = to_categorical(encoded_categories)

# Делим данные на 4 категории:
# Резюме для тестирования, резюме для обучения,
# Категории для тестирования, категории для обучения
X_train, X_test, y_train, y_test = train_test_split(
    all_resume_values_vectors,
    encoded_categories_matrix,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=encoded_categories
)

# Создаем нейроны (слои)
first_layer = Dense(128, activation='relu', input_dim=all_resume_values_vectors.shape[1])
second_layer = Dense(64, activation='relu')
third_layer = Dense(y_train.shape[1], activation='softmax')

# Создаем нейронную сеть
model = Sequential()

# Добавляем слои
model.add(first_layer)
model.add(second_layer)
model.add(third_layer)

# Компилируем модель
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучаем модель
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Получаем точность и потери на тестовой выборке
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Точность нейронной сети на тестовой выборке: {accuracy:.2f}')
print(f'Расчет количества потерь на тестовой выборке: {loss:.2f}')

from sklearn.metrics import classification_report

# Получаем отчет о результатах на тестовой выборке
results = model.predict(X_test)
found_categories = np.argmax(results, axis=1)
true_categories = np.argmax(y_test, axis=1)
result_report = classification_report(true_categories, found_categories, target_names=label_encoder.classes_)
print(f"Отчет: \n{result_report}")

# Получаем замеры времени выполнения
end_time = time.time()
print(f"Время на обучение: {end_time - start_time} секунд")
