from textblob import TextBlob
import spacy

nlp = spacy.load('en_core_web_sm')

def clear_html_characters(html_text: str) -> str:
    return html_text.str.replace("(<br/>)", "", regex=True
                        ).replace('(<a).*(>).*(</a>)', '', regex=True
                        ).replace('(&amp)', '', regex=True
                        ).replace('(&gt)', '', regex=True
                        ).replace('(&lt)', '',regex=True
                        ).replace('(\xa0)', ' ',regex=True)


dependency_to_delete = ['NUM', 'INTJ', 'CONJ', 'ADV',
                        'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'PRON', 'SYM', 'x']


def spacy_clean(text):
    spacy_text = ""
    doc = nlp(text)
    for token in doc:
        if token.is_stop == False and token.is_alpha and len(token) > 2 and token.pos_ not in dependency_to_delete:
            spacy_text += token.lemma_ + " "
    return spacy_text.rstrip()


def add_clean_text_columns(dataframe, text_column="text"):
    # spacy function that removes unwanted words from Twitter posts
    dataframe["clean_text"] = clear_html_characters(
        dataframe[text_column].str.lower())
    dataframe['clean_text'] = dataframe['clean_text'].apply(
        lambda x: spacy_clean(x))
    dataframe = dataframe.dropna()
    dataframe['polarity_tokens'] = dataframe['clean_text'].map(
        lambda text: TextBlob(text).sentiment.polarity)
    dataframe['review_len'] = dataframe['clean_text'].astype(str).apply(len)
    dataframe['word_count'] = dataframe['clean_text'].apply(
        lambda word: len(str(word).split()))
    return dataframe