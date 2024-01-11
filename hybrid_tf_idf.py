import nltk
import os
import re
import math
import operator
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
Stopwords = set(stopwords.words('indonesian'))
wordlemmatizer = WordNetLemmatizer()

def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words

def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex, '', text)
    return text

def custom_stopwords_removal(words):
    additional_stopwords = [
        "yg", "dg", "rt", "dgn", "ny", "d", 'klo',
        'kalo', 'amp', 'biar', 'bikin', 'bilang',
        'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
        'jd', 'jgn', 'sdh', 'aja', 'n', 't',
        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
        '&amp', 'yah', 'bla', 'wkwk', 'Puji'
    ]
    nltk_stopwords = set(stopwords.words('indonesian'))
    list_stopwords = set(nltk_stopwords).union(additional_stopwords)
    return list_stopwords

def stopwords_removal(words, list_stopwords):
    clean_words = [word for word in words if word.lower() not in list_stopwords]
    return clean_words

def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
           words_unique.append(word)
    for word in words_unique:
       dict_freq[word] = words.count(word)
    return dict_freq

def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word, tag in pos_tag:
        if tag.startswith("NN") or tag.startswith("VB"):
             pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb

def tf_score(word, sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf = word_frequency_in_sentence / len_sentence
    return tf

def idf_score(no_of_sentences, word, sentences, list_stopwords):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        clean_sentence = remove_special_characters(str(sentence))
        clean_sentence = re.sub(r'\d+', '', clean_sentence)
        clean_sentence = clean_sentence.split()
        clean_sentence = [word for word in clean_sentence if word.lower() not in list_stopwords and len(word) > 1]
        clean_sentence = [word.lower() for word in clean_sentence]
        clean_sentence = [wordlemmatizer.lemmatize(word) for word in clean_sentence]
        if word in clean_sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences / (no_of_sentence_containing_word + 1))
    return idf

def tf_idf_score(tf, idf):
    return tf * idf

def hybrid_tfidf_score(tf, idf, sentence_position_score):
    return tf * idf * sentence_position_score

def sentence_position_score(sentence_index, total_sentences):
    return 1 - (sentence_index / total_sentences)

def word_tfidf(dict_freq, word, sentence, list_stopwords):
    tf = tf_score(word, sentence)
    idf = idf_score(len(sentence), word, sentence, list_stopwords)
    tf_idf = tf_idf_score(tf, idf)
    return tf_idf

def sentence_importance(sentence, dict_freq, sentences, list_stopwords):
    sentence_score = 0
    clean_sentence = remove_special_characters(str(sentence))
    clean_sentence = re.sub(r'\d+', '', clean_sentence)
    pos_tagged_sentence = pos_tagging(clean_sentence)
    for word in pos_tagged_sentence:
        if word.lower() not in list_stopwords and word not in list_stopwords and len(word) > 1:
            word = word.lower()
            word = wordlemmatizer.lemmatize(word)
            sentence_score = sentence_score + word_tfidf(dict_freq, word, clean_sentence, list_stopwords)
    return sentence_score

def run_hybrid_tf_idf(text):
    tokenized_sentence = sent_tokenize(text)
    text = remove_special_characters(text)
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    Stopwords = set(stopwords.words("indonesian"))  # Ganti dengan stopwords bahasa Indonesia
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words)
    word_freq = freq(tokenized_words)
    list_stopwords = custom_stopwords_removal(tokenized_words)
    clean_words = stopwords_removal(tokenized_words, list_stopwords)

    # input_user = float(input('Persentase peringkasan teks:'))
    input_user = 30
    no_of_sentences = int((input_user * len(tokenized_sentence)) / 100)
    #print(f"Jumlah Kalimat dalam Ringkasan : {no_of_sentences} kalimat")

    c = 1
    sentence_with_importance = {}
    total_sentences = len(tokenized_sentence)

    for sent in tokenized_sentence:
        sentence_imp = sentence_importance(sent, word_freq, tokenized_sentence, list_stopwords)
        sentence_pos_score = sentence_position_score(c, total_sentences)
        hybrid_score = hybrid_tfidf_score(sentence_imp, 1.5, sentence_pos_score)
        sentence_with_importance[c] = hybrid_score
        c = c + 1

    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1), reverse=True)
    cnt = 0
    summary = []
    sentence_no = []

    for word_prob in sentence_with_importance:
        if cnt < no_of_sentences:
            sentence_no.append(word_prob[0])
            cnt = cnt + 1
        else:
            break

    sentence_no.sort()
    cnt = 1

    for sentence in tokenized_sentence:
        if cnt in sentence_no:
            summary.append(sentence)
        cnt = cnt + 1

    summary = " ".join(summary)
    return summary