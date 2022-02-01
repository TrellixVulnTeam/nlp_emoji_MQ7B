import spacy
import pandas as pd
import flag


def get_locations(document):  # Получает span локаций
    locations = []
    # Поиск локаций по лейблу LOC
    for ent in document.ents:
        if ent.label_ == "LOC":
            locations.append(ent)
    if locations:
        return lemmatize(locations)
    else:
        return None


def lemmatize(entities):  # Приводит к инфинитиву локацию
    lemmatized = []
    for entity in entities:
        lem_entity = ''
        # Получение списка токенов которые надо привести к инфинитиву
        span_entity = entity.doc[entity.start:entity.end]
        for word in span_entity:
            lem_entity += word.lemma_
        lemmatized.append(lem_entity)
    if lemmatized:
        return lemmatized
    else:
        return None


def get_alpha2(locations):  # Получает код страны по названию страны
    alphas = []
    for location in locations:
        location = location.title()
        row_ids = []
        # Поиск страны по таблице
        for row_id in countries.index[countries['name'] == location].tolist():
            row_ids.append(row_id)
        for row_id in countries.index[countries['fullname'] == location].tolist():
            row_ids.append(row_id)

        for row_id in row_ids:
            alphas.append(countries['alpha2'][row_id])

    if alphas:
        return alphas
    else:
        return None


countries = pd.read_csv('country-database/countries.csv', sep=';', encoding='cp1251')  # Загрузка таблицы стран и их кодов
nlp = spacy.load("ru_core_news_sm")

sentence = "Сборная России по фигурному катанию прибыла в Пекин для участия в Олимпийских играх, сообщает «Матч ТВ»."
doc = nlp(sentence)

locs = get_locations(doc)
if locs:
    alphs = get_alpha2(locs)
    flags = list(map(flag.flag, alphs))  # Конвертация кода страны в эмоджи
    print("".join(flags) + " " + sentence)
