import math
from nltk import sent_tokenize, word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation
import heapq
from wiki_ru_wordnet import WikiWordnet


# Функция подсчета частоты слов в каждом предложении.
def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("russian"))
    ps = SnowballStemmer("russian")
    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent, language='russian')

        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords or word in punctuation:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


# Функция подсчета TermFrequency для каждого слова в абзаце.
# TF (t) = (Количество раз, когда термин t появляется в документе) / (Общее количество терминов в документе)
# Вывод: документ - это абзац, термин - это слово в абзаце.
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


# Функция подсчета количества предложений в которых участвуют слова.
# Простая таблица, которая помогает в расчете IDF-матрицы.
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


# Функция подсчета IDF для каждого слова в абзаце.
# IDF (t) = log_e (общее количество документов / количество документов с термином t в нем)
# Обратная частота документа (IDF) показывает, насколько уникальным или редким является слово.
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


# Функция подсчета TF-IDF
# Алгоритм TF-IDF состоит из двух алгоритмов, умноженных вместе.
# Происходит умножение значений матриц и генерация новой.
def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


# Функция подсчета веса предложения (оценка)
# Здесь используется Tf-IDF количества слов в предложении
# Алгоритм: сложение частоты TF каждого слова в предложении, деленной на общее количество слов в предложении.
def _score_sentences(tf_idf_matrix) -> dict:
    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


def _score_sentences_back(tf_idf_matrix) -> dict:
    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[total_score_per_sentence / count_words_in_sentence] = sent

    return sentenceValue


# Функция подсчета среднего балла за предложение
def _find_average_score(sentenceValue) -> float:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Среднее значение предложения из исходного summary_text
    average = (sumValues / len(sentenceValue))

    return average


# Функция разделения текста на предложения с большим весом и резерв
def _fragmentation_sentences(sentences, sentenceValue, maximum, average, minimum):
    abstract = ''
    reserve = ''

    upper_threshold = (maximum - average) / 2 + average
    lower_threshold = (average - minimum) / 2 + minimum

    # for sentence in sentences:
    #     if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= upper_threshold:
    #         abstract += sentence + " "
    #
    # for sentence in sentences:
    #     if sentence[:15] in sentenceValue and upper_threshold > sentenceValue[sentence[:15]] >= lower_threshold:
    #         reserve += sentence + " "

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= average:
            abstract += sentence + " "

    for sentence in sentences:
        if sentence[:15] in sentenceValue and average > sentenceValue[sentence[:15]] >= minimum:
            reserve += sentence + " "

    return abstract, reserve


# Функция добавления из резерва недостающего количества предложений
def _generate_reserve(reserve, sentenceValue, length_missing):
    sentence_count = 0
    summary = ''

    dict_max = heapq.nlargest(length_missing, sentenceValue.items())
    list_max = []
    for elem in dict_max:
        list_max.append(elem[1])

    for sentence in reserve:
        if sentence[:15] in list_max:
            summary += sentence + " "
            sentence_count += 1

    return summary


def _add_reserve(reserve_sentence, length_missing_sentences):
    total_documents = len(reserve_sentence)
    # Создание матрицы частоты слов в каждом предложении
    freq_matrix = _create_frequency_matrix(reserve_sentence)
    # Рассчитать TermFrequency и создать ее матрицу
    tf_matrix = _create_tf_matrix(freq_matrix)
    # Создание таблицы по словам текста
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    # Вычисление IDF и генерация матрицы
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    # Подсчет TF-IDF и генерация матрицы
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    # Оценка предложения
    sentence_scores = _score_sentences_back(tf_idf_matrix)

    summary = _generate_reserve(reserve_sentence, sentence_scores, length_missing_sentences)
    summary = sent_tokenize(summary, language='russian')

    return summary


# Функция составления финального результата с резервом
def _generate_result(main_sentence, summary, sentences):
    total = ''
    for sentence in sentences:
        if sentence in main_sentence or sentence in summary:
            total += sentence + " "
    return total


# Функция исключения из финальной версии предложений с большим количеством синонимов
def _del_synonyms(sentences) -> dict:
    stopWords = set(stopwords.words("russian"))
    wikiwordnet = WikiWordnet()
    synonyms_matrix = {}
    ps = SnowballStemmer("russian")

    for sent in sentences:
        words = word_tokenize(sent, language='russian')
        synonyms = []
        summ = []

        for word in words:
            word = word.lower()

            if word in stopWords or word in punctuation:
                continue

            summ.append(ps.stem(word))

            synsets = wikiwordnet.get_synsets(word)

            if synsets:
                for w in synsets[0].get_words():
                    synonyms.append(ps.stem(w.lemma()))
            else:
                synonyms.append(word)

        synonyms_table = {}

        for syn in synonyms:
            if syn in summ:
                if synonyms.count(syn) > 1:
                    synonyms_table[syn] = synonyms.count(syn)
                else:
                    synonyms_table[syn] = 1

        synonyms_matrix[sent[:15]] = synonyms_table

    return synonyms_matrix


def _sample_sentences(synonyms_matrix, main_sentence):
    sentence_result = ''

    for sent, s_table in synonyms_matrix.items():
        k = 0
        for word, score in s_table.items():
            if score > 1:
                k += 1
        if k == 0:  # синонимов нет
            for sentence in main_sentence:
                if sent == sentence[:15]:
                    sentence_result += sentence + " "

    return sentence_result


# Запись в файл
def _writing_file(result, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(result)
    file.close()


# Запись сводки
def _writing_resume(result, text):
    with open("resume.txt", "a", encoding="utf-8") as file_resume:
        file_resume.write(text + "\n" + str(result) + "\n\n")
    file_resume.close()


def ACSI_Matic(sentences, total_documents):
    # Подсчет размера сжатого текста
    abstract_size = int(total_documents / 10 + 0.5)
    _writing_resume(abstract_size, "Объем сжатого текста:")

    # Создание матрицы частоты слов в каждом предложении
    freq_matrix = _create_frequency_matrix(sentences)
    _writing_resume(freq_matrix, "Матрица частоты слов в предложениях:")

    # Создание матрицы частоты слов во всем тексте
    tf_matrix = _create_tf_matrix(freq_matrix)
    _writing_resume(tf_matrix, "Матрица частоты слов в тексте:")

    # Создание таблицы по словам текста
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    _writing_resume(count_doc_per_words, "Таблица - 'сколько предложений содержат слово ***':")

    # Вычисление IDF и генерация матрицы
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    _writing_resume(idf_matrix, "Таблица обратной частоты текста:")

    # Подсчет TF-IDF и генерация матрицы
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    _writing_resume(tf_idf_matrix, "Результирующая таблица двух предыдущих:")

    # Оценка предложения
    sentence_scores = _score_sentences(tf_idf_matrix)
    _writing_resume(sentence_scores, "Вес каждого предложения:")

    # Поиск порога
    threshold = _find_average_score(sentence_scores)
    max_val = max(sentence_scores.values())
    min_val = min(sentence_scores.values())
    _writing_resume(threshold, "Средний вес среди всех предложений:")
    _writing_resume(max_val, "Максимальный вес среди всех предложений:")
    _writing_resume(min_val, "Минимальный вес среди всех предложений:")

    # Фрагментация текста на главные предложения и резервные
    main_sentence, reserve_sentence = _fragmentation_sentences(sentences, sentence_scores, max_val, threshold, min_val)
    _writing_file(main_sentence, "main_sentence.txt")
    _writing_file(reserve_sentence, "reserve_sentence.txt")

    main_sentence = sent_tokenize(main_sentence, language='russian')
    reserve_sentence = sent_tokenize(reserve_sentence, language='russian')
    main_sentence_size = len(main_sentence)

    _writing_resume(main_sentence_size, "Объем сжатого текста:")

    # Если объем сжатого реферата меньше требуемого значения
    if main_sentence_size < abstract_size:
        length_missing_sentences = abstract_size - main_sentence_size
        _writing_resume(length_missing_sentences,
                        "Объем сжатого реферата меньше требуемого значения. Количество не хватающих предложений:")
        summary = _add_reserve(reserve_sentence, length_missing_sentences)

        # Создание нового сжатого текста, на основании отбора из резерва и основного
        _writing_file(_generate_result(main_sentence, summary, sentences), "final_text.txt")
        return

    # Если объем сжатого реферата больше требуемого значения
    if main_sentence_size > abstract_size:
        _writing_resume(main_sentence_size - abstract_size,
                        "Объем сжатого реферата больше требуемого значения. Количество лишних предложений:")
        # Генерация матрицы количества синонимов для каждого слова в предложениях
        synonyms_matrix = _del_synonyms(main_sentence)
        _writing_resume(synonyms_matrix,
                        "Таблица количества синонимов для слов в предложениях:")
        # Выборка предложений без синонимов
        sample_sentences = _sample_sentences(synonyms_matrix, main_sentence)

        _writing_file(sample_sentences, "final_text.txt")

        sample_sentences = sent_tokenize(sample_sentences, language='russian')
        length_sample_sentences = len(sample_sentences)

        if length_sample_sentences > abstract_size:  # обработка методом ACSI_Matic сжатого текста
            _writing_resume(length_sample_sentences - abstract_size,
                            "Объем сжатого реферата больше требуемого значения. Количество лишних предложений:")
            ACSI_Matic(main_sentence, total_documents)

        if length_sample_sentences < abstract_size:  # добавление из резерва
            length_missing_sentences = abstract_size - length_sample_sentences
            _writing_resume(length_missing_sentences,
                            "Объем сжатого реферата меньше требуемого значения. Количество не хватающих предложений:")
            total_sample_sentences = _add_reserve(reserve_sentence, length_missing_sentences)

            # Создание нового сжатого текста, на основании отбора из резерва и основного
            result = _generate_result(main_sentence, total_sample_sentences, sentences)

            _writing_file(result, "final_text.txt")
            return

    # Объем сжатого реферата соответствует требуемому значению
    if main_sentence_size == abstract_size:
        _writing_resume(abstract_size,
                        "Объем сжатого реферата, как и требовалось:")
        with open("main_sentence.txt", "r", encoding="utf-8") as file_main:
            _writing_file(file_main.read(), "final_text.txt")
        return


if __name__ == '__main__':
    with open("original_text.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # Токенизация предложений
    sentences = sent_tokenize(text, language='russian')
    total_documents = len(sentences)

    with open("resume.txt", "w", encoding="utf-8") as file:
        file.write("Сводка" + "\n")
    file.close()

    _writing_resume(sentences, "Разбиение текста на предложения:")
    _writing_resume(total_documents, "Общее количество предложений:")

    ACSI_Matic(sentences, total_documents)
