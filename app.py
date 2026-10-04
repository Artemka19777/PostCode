import joblib
import numpy as np
import pandas as pd
import streamlit as st
from text_utils import clean2
import sqlite3
import datetime
from contextlib import closing

DB = 'tickets.db'

def init_db():
    with closing(sqlite3.connect(DB)) as conn, conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS new_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT, text TEXT, service TEXT, component TEXT, client TEXT,
            category TEXT, confidence REAL, line TEXT, line_confidence REAL)''')

def save_ticket(text, service, comp, client, category, conf, line, line_conf):
    with closing(sqlite3.connect(DB)) as conn, conn:
        conn.execute(
            '''INSERT INTO new_tickets
               (created, text, service, component, client, category, confidence, line, line_confidence)
               VALUES (?,?,?,?,?,?,?,?,?)''',
            (datetime.datetime.now().isoformat(timespec='seconds'), text, service, comp, client,
             category, float(conf), line, float(line_conf)))
st.set_page_config(page_title='Service Desk', layout='wide')

init_db()

@st.cache_resource
def load():
    return (joblib.load('models/model_category_v1.joblib'), joblib.load('models/model_line_v1.joblib'),
            joblib.load('models/line_history.joblib'), joblib.load('models/options.joblib'))

cat_model, line_model, line_hist, options = load()
LINE_COL = 'Кем решен (группа)'

tab1, tab2 = st.tabs(['Новое обращение','Журнал'])

with tab1:
    st.title('Первичная обработка обращения')
    text = st.text_area('Описание обращения', height=180)
    c1, c2, c3 = st.columns(3)
    service = c1.selectbox('Услуга', options['services'])
    comp = c2.selectbox('Компонент услуги 1 уровня', options['components'])
    user = c3.text_input('Пользователь', 'Не указан')

    if st.button('Определить') and text.strip():
        row = pd.DataFrame([{'text': clean2(text), 'Услуга': service,
                             'Компонент услуги 1 уровня': comp,
                             'Пользователь': user or 'Не указан'}])

        # --- вид обращения ---
        proba = cat_model.predict_proba(row)[0]
        order = proba.argsort()[::-1][:3]
        best, conf = cat_model.classes_[order[0]], proba[order[0]]

        st.subheader('Вид обращения')
        if conf >= 0.7:
            st.success(f'{best} (уверенность {conf:.0%}): уверенно')
        elif conf >= 0.5:
            st.warning(f'{best} (уверенность {conf:.0%}): проверьте рекомендацию')
        else:
            st.error(f'Недостаточно уверен (лучший вариант: {best}, {conf:.0%}). '
                     'Передайте оператору для ручного разбора.')
        st.caption('Уверенность это вероятность, которую выдаёт модель для лучшего из 15 видов. '
                   'Модель знает только 15 основных видов; обращения других видов могут получить '
                   'ответ, поэтому при низкой уверенности нужна ручная проверка.')
        st.table(pd.DataFrame({'Вид': cat_model.classes_[order],
                               'Вероятность': [f'{p:.0%}' for p in proba[order]]}))

        # --- объяснение: слова, которые повлияли на выбор ---
        try:
            prep, clf = cat_model.named_steps['prep'], cat_model.named_steps['clf']
            names = prep.get_feature_names_out()
            x = prep.transform(row)
            contrib = x.multiply(clf.coef_[order[0]]).toarray().ravel()
            words = [names[i].split('__')[-1] for i in np.argsort(-contrib)[:40]
                     if contrib[i] > 0 and 'tfidfvectorizer-1' in names[i]][:6]
            if words:
                st.write('**Почему так:** на выбор повлияли слова и выражения: ' + ', '.join(words))
        except Exception:
            pass

        # --- линия поддержки ---
        lp = line_model.predict_proba(row)[0]
        li = lp.argmax()
        save_ticket(text, service, comp, user or 'Не указан',
                    best, conf, line_model.classes_[li], lp[li])
        st.subheader('Рекомендуемая линия')
        st.metric(line_model.classes_[li], f'{lp[li]:.0%}')
        share = (line_hist[line_hist['Вид запроса'] == best][LINE_COL]
                 .fillna('Не указана').value_counts(normalize=True))
        st.write(f'**Почему так:** в истории для вида «{best}» обращения решали: ' +
                 ', '.join(f'{k} {v:.0%}' for k, v in share.items()) +
                 '. Рекомендация учитывает также текст, услугу и компонент.')

with tab2:
    st.subheader('Введённые обращения')
    with closing(sqlite3.connect(DB)) as conn:
        log = pd.read_sql('SELECT * FROM new_tickets ORDER BY id DESC LIMIT 100', conn)
    st.dataframe(log, use_container_width=True)