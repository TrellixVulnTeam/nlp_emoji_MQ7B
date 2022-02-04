import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import numpy as np
import flag
import emoji


def get_locations(document):  # Получает span локаций
    locations = []
    # Поиск локаций по лейблу LOC
    for ent in document.ents:
        if ent.label_ == "LOC":
            locations.append(ent)
    if locations:
        return lemmatize(locations)


def lemmatize(entities):  # Приводит к инфинитиву локацию
    lemmatized = []
    for entity in entities:
        lem_entity = ''
        # Получение списка токенов которые надо привести к инфинитиву
        span_entity = entity.doc[entity.start:entity.end]
        for word in span_entity:
            lem_entity += word.lemma_ + " "
        lemmatized.append(lem_entity.strip())
    if lemmatized:
        return lemmatized


def get_alpha2(locations):  # Получает код страны по названию страны
    alphas = []
    for location in locations:
        location = location.title()
        # Поиск страны по таблице
        for row_id in countries_table.index[np.logical_or(countries_table['name'] == location,
                                                    countries_table['fullname'] == location)].tolist():
            alphas.append(countries_table['alpha2'][row_id])
    if alphas:
        return alphas


def get_locations_emoji(document):  # Получает эмодзи стран из Doc
    locs = get_locations(document)
    if locs:
        alphs = get_alpha2(locs)
        flags = list(map(flag.flag, alphs))  # Конвертация кода страны в эмоджи
        return flags


def get_matched_sports(document):  # Находит виды спорта совпадающие с паттерном
    sports = []  # Список видов спорта
    for sport in sports_table['name']:
        sports.append(sport)
    # TODO: Чувствителен к регистру
    matcher = PhraseMatcher(document.vocab, attr="LEMMA")
    patterns = [nlp(sport) for sport in sports]  # Добавление видов спорта в паттерн
    matcher.add("SportList", patterns)  # Добавление паттерна в PhraseMatcher
    matches = lemmatize(list(set(matcher(document, as_spans=True))))  # Поиск в Doc совпадений с паттерном
    if matches:
        return matches


def get_sport_emojis(document):  # Получает эмодзи видов спорта из Doc
    emojis = []
    matched_sports = get_matched_sports(document)
    if matched_sports:
        for sport in matched_sports:
            row_id = sports_table.index[sports_table['name'] == sport][0]  # Поиск вида спорта в таблице
            emojis.append(sports_table['emoji'][row_id])  # Добавление эмодзи вида спорта
        if emojis:
            return list(map(str.strip, map(emoji.emojize, emojis)))  # Конвертация в эмодзи и вывод


def main():
    global countries_table
    countries_table = pd.read_csv('country-database/countries.csv', sep=';', encoding='cp1251')  # Загрузка таблицы стран и их кодов

    global sports_table
    sports_table = pd.read_csv('sports-database/sports.csv', sep=';', encoding='cp1251') # Загрузка таблицы видов спорта и кодов эмодзи

    global nlp
    nlp = spacy.load("ru_core_news_sm")
    sentence = "Сборная России, Катара, Румынии и Германии по фигурному катанию и бадминтону прибыла в " \
               "Пекин для участия в Олимпийских играх, сообщает «Матч ТВ»."

    doc = nlp(sentence)
    loc_emojis = get_locations_emoji(doc)

    sport_emojis = get_sport_emojis(doc)
    print("".join(loc_emojis) + "".join(sport_emojis) + " " + sentence)


main()
