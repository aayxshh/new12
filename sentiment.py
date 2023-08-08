import pandas as pd
import streamlit as st
import cleantext
from textblob import TextBlob

def score(x):
    blob = TextBlob(x)
    return blob.sentiment.polarity

def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    st.header('Sentiment Analysis')

    with st.expander('Analyze Text'):
        text = st.text_input('Text here: ')
        if text:
            blob = TextBlob(text)
            st.write('Polarity: ', round(blob.sentiment.polarity, 2))
            st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))

    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                 stopwords=True, lowercase=True, numbers=True, punct=True))

    with st.expander('Analyze CSV'):
        upl = st.file_uploader('Upload file')

        if upl:
            df = pd.read_excel(upl)
            del df['Unnamed: 0']
            df['score'] = df['tweets'].apply(score)
            df['analysis'] = df['score'].apply(analyze)
            st.write(df.head(10))

            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
