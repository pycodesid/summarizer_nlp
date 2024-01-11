# Libraries

# Streamlit
import streamlit as st

# Utilities
import os
import subprocess
import threading
import sys
import importlib

# Dataframe & Visualization
import pandas as pd
import matplotlib.pyplot as plt

# Math
import numpy as np
import math
import operator

# Text Preprocessing
import re
from io import StringIO
from PyPDF2 import PdfReader
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

# Import From Other Python File
from preprocessing_m import preprocess # preprocessing texts
from CLDA import run_clda # running CLDA algorithm
from bert import run_bert # running bert algorithm
from hybrid_tf_idf import run_hybrid_tf_idf # running hybrid tf idf algorithm
from tf_idf import run_tf_idf # running tf idf algorith
from rouge_metric import run_rouge_metric # running rouge metric algorithm


# Extract Text From PDF File
def extract_text_from_pdf(file):
    pdf_text = ""
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)

    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()

    return pdf_text


def remove_name_and_date(text):
    # time_pattern = r'\s*\(.*?\)|\w+\s*,\s+\d+\s+\w+\s+\d+\s+\d+:\d+:\d+\s*'
    # cleaned_text1 = re.sub(time_pattern, '', text)
    split_text = re.split('#', text)
    split_text_new = split_text[1:len(split_text)-1]
    cleaned_text2 = '\n'.join(split_text_new)

    return cleaned_text2


def word_page_count(text, search):
    list_page = []
    pages = text.split("\f")
    for i, page_text in enumerate(pages):
        if re.findall(search, page_text):
            count_page = len(re.findall(search, page_text))
            list_page.append((count_page, i + 1))

    count = len(list_page)
    total = sum([tup[0] for tup in list_page])

    return total, count


def split_into_sentences(text):
    sentences = text.split('. ')
    return sentences


def save_results(extracted_text, save_as):
    if save_as == "DataFrame":
        df = pd.DataFrame({"Text": [extracted_text]})
        st.write(df)
    elif save_as == "File .txt":
        with open("extracted_text.txt", "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)
        st.success("Teks berhasil disimpan dalam file extracted_text.txt")


def save_to_txt(file_name, text):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    st.title(':bookmark_tabs: :blue[ALRISA]')
    st.header('_:blue[Aplikasi Peringkas Tesk Otomatis]_')
    st.write("""   
    Aplikasi ini merupakan instrumen yang dapat digunakan untuk mengekstraksi informasi dari file Dokumen Forum Penelaahan. Hasil ekstraksi berupa permodelan topik dan ringkasan dari file yang diunggah. Aplikasi dikembangkan dengan menggunakan model C-LDA dan Hybrid-TF-IDF.""")
   
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Tahap Preprocessing:", ["Ekstraksi", "Cari Kata", "Preprocessing"])
    menu2 = st.sidebar.selectbox("Peringkasan Teks:", ["Permodelan", "Ringkasan"])
    menu3 = st.sidebar.selectbox("Evaluasi Model:", ["ROUGE Metric", "Perplexities"])
    
    # Session initialization
    if 'extracted_texts' not in st.session_state:
        st.session_state.extracted_texts = []

    if 'cleaned_texts' not in st.session_state:
        st.session_state.cleaned_texts = []

    if 'merged_text' not in st.session_state:
        st.session_state.merged_text = []

    if 'merged_text2' not in st.session_state:
        st.session_state.merged_text2 = []

    if 'text4' not in st.session_state:
        st.session_state.text4 = []
    
    if 'stage1' not in st.session_state:
        st.session_state.stage1 = 0

    if 'stage2' not in st.session_state:
        st.session_state.stage2 = 0

    if 'txt_upload' not in st.session_state:
        st.session_state.txt_upload = []

    if 'proses_ringkas' not in st.session_state:
        st.session_state.proses_ringkas = 0

    if 'hasil_ringkas' not in st.session_state:
        st.session_state.hasil_ringkas = []

    if 'hasil_rouge_metric' not in st.session_state:
        st.session_state.hasil_rouge_metric = []

    if 'file_ref' not in st.session_state:
        st.session_state.file_ref = []

    if 'butt_r' not in st.session_state:
        st.session_state.butt_r = 0

    if 'file_name_rouge' not in st.session_state:
        st.session_state.file_name_rouge = []

    if 'text4_upload' not in st.session_state:
        st.session_state.text4_upload = []

    if 'merged_text2_upload' not in st.session_state:
        st.session_state.merged_text2_upload = []

    if 'text4_' not in st.session_state:
        st.session_state.text4_ = []

    if 'merged_text2_' not in st.session_state:
        st.session_state.merged_text2_ = []


    if menu == "Ekstraksi":
        st.write("Unggah file PDF untuk mengekstraksi file.")

        uploaded_files = st.file_uploader("Pilih file PDF", type=["pdf"], accept_multiple_files=True)
        if uploaded_files and st.button("Ekstrak File"):
            for pdf_file in uploaded_files:
                extracted_text = extract_text_from_pdf(pdf_file)
                st.session_state.extracted_texts.append(extracted_text)

            st.success("Ekstraksi teks berhasil!")

            st.subheader("Hasil Ekstraksi")
            for idx, extracted_text in enumerate(st.session_state.extracted_texts):
                cleaned_text = remove_name_and_date(extracted_text)
                st.session_state.cleaned_texts.append(cleaned_text)
   
                st.write(f"Hasil Ekstraksi File {idx + 1}:")
                st.text_area(f"Hasil Ekstraksi File {idx + 1}", value=cleaned_text, height=300)

        if len(st.session_state.cleaned_texts) and st.button("Gabung Hasil Ekstraksi"):
            st.session_state.merged_text = ["\n".join(st.session_state.cleaned_texts)]

            st.success("Gabungan Hasil Ekstraksi teks berhasil!")

            st.subheader("Gabungan Hasil Ekstraksi")
            st.text_area("Gabungan Hasil Ekstraksi", value=st.session_state.merged_text[0], height=300)
            save_to_txt('input01.txt', st.session_state.merged_text[0])

        if st.button("Jangan Gabung Hasil Ekstraksi"):
            st.success("Hasil Ekstraksi Tetap Terpisah!")
            st.session_state.merged_text = st.session_state.cleaned_texts


    if menu == "Cari Kata":
        st.write("Cari kata spesifi dari file yang diunggah")

        if 'search' not in st.session_state:
            st.session_state.search = ''

        # Kotak input untuk mencari kata
        st.session_state.search = st.text_input('Masukkan kata yang dicari:')
        if not st.session_state.search:
            st.info("Silakan memasukkan kata yang akan dicari")
            return

        # Cari kata
        if st.session_state.extracted_texts is None:
            st.warning("Silakan unggah file PDF terlebih dahulu pada menu 'Ekstraksi'.")
        else:
            for idx, extracted_text in enumerate(st.session_state.extracted_texts):
                total, count = word_page_count(extracted_text, st.session_state.search)
                st.subheader(f"Kata '{st.session_state.search}' ditemukan {total} kali pada {count} halaman pada dokumen ke - {idx+1}")

    elif menu == "Preprocessing":
            if len(st.session_state.text4) == 0:
                for idx, m_text in enumerate(st.session_state.merged_text):  
                    text2 = preprocess(m_text)

                    st.session_state.text4.append(text2)

                    text3 = []
                    for item in text2:
                        text3.append(" ".join(item))
                    text3 = " ".join(text3)
                    st.session_state.merged_text2.append(text3) 

                    st.subheader(f"Hasil Preprocessing Teks")
                    st.text_area(f"Hasil Preprocessing Teks - {idx+1}", value=text3, height=300)

         # Tampilkan pilihan untuk menyimpan hasil ekstraksi dalam bentuk DataFrame atau file .txt
            st.subheader("Pilih bentuk penyimpanan hasil preprocessing:")
            save_option = st.radio("", ["DataFrame", "File .txt"])
            if save_option == "File .txt":
                file_name = st.text_input("Masukkan nama file untuk disimpan (.txt):", value="hasil_ekstraksi.txt")
                file_name = file_name.strip()

                if file_name.endswith(".txt") and st.button("Simpan"):
                    i = st.session_state.get("file_count", 1)
                    for j in range(len(st.session_state.merged_text2)):
                        save_file_name = f"File_Ekstraksi_hasil_{j}_versi_{i}.txt"
                        save_to_txt(save_file_name, st.session_state.merged_text2[j])  # Use merged_text instead of cleaned_text
                    st.success(f"Sukses menyimpan Hasil Ekstraksi File '{save_file_name}'.")
                    st.session_state.file_count = i + 1

            elif save_option == "DataFrame":
                if st.button("Simpan"):
                    df = pd.DataFrame({"Text": st.session_state.merged_text2})  # Use merged_text instead of cleaned_text
                    st.dataframe(df)

            st.success("Preprocessing berhasil!")
            st.subheader("Ingin menggunakan hasil preprocessing untuk pemodelan atau upload?")

            button_lanjut = st.button("Lanjutkan")
            button_upload = st.button("Upload")

            if button_lanjut:
                st.session_state.stage1 = 1
                st.session_state.stage2 = 0

            if button_upload:
                st.session_state.stage2 = 1
                st.session_state.stage1 = 0


    if menu2 == "Permodelan"  and st.session_state.stage1 == 1:
        start = 5
        end = 1
        iteration_num = 5
        clip = 50
        palpha = 0.05
        pgamma = 0.05
        pbeta = 0.05
        c_len = 10
        save_p = "Contextual_LDA_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"

        if len(st.session_state.text4) > 0:
            for idx, textt in enumerate(st.session_state.text4):
                print(f"Lanjutkan {idx}")
                result = run_clda(textt, start, end, iteration_num, save_p, clip, c_len, palpha, pbeta, pgamma)
                if result is not None:
                    st.dataframe(result[1])
                    doc_lens = [len(d) for d in result[1].Text]

                    # Plot
                    plt.figure(figsize=(8,5), dpi=160)
                    plt.hist(doc_lens, bins = 35, color='navy')
                    plt.text(120, 4, "Mean   : " + str(round(np.mean(doc_lens))))
                    plt.text(120, 3.5, "Median : " + str(round(np.median(doc_lens))))
                    plt.text(120, 3, "Stdev   : " + str(round(np.std(doc_lens))))
                    plt.text(120, 2.5, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
                    plt.text(120, 2, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

                    plt.gca().set(xlim=(0, 100), ylabel='Number of Documents', xlabel='Document Word Count')
                    plt.tick_params(size=16)
                    plt.xticks(np.linspace(0,150,9))
                    plt.title('Distribution of Document Word Counts', fontdict=dict(size=20))
                    st.pyplot(plt.gcf())

                    from wordcloud import WordCloud
                    # define subplot grid
                    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(15, 12))
                    plt.subplots_adjust(hspace=0.5)
                    fig.suptitle("Word Cloud Dominant Topic Words", fontsize=18, y=0.95)
                    
                    p_df = result[1].copy()
                    new_ = p_df.groupby(['dominant_topic'], as_index = False).agg({'sentence': ' '.join})

                    # loop through tickers and axes
                    for item, i, ax in zip(new_.sentence, new_.dominant_topic, axs.ravel()):
                        word_cloud = WordCloud(collocations = False, background_color = 'white').generate(item)
                        ax.imshow(word_cloud, interpolation='bilinear')
                        ax.set_title(f"Topic - {i}".upper())
                        ax.set_xlabel("")
                    st.pyplot(plt.gcf())
                    # st.session_state.stage1 = 0
                else:
                    st.error("Error")


    if menu2 == "Permodelan"  and st.session_state.stage2 == 1:
        # st.session_state.stage1 = 0
        start = 5
        end = 1
        iteration_num = 5
        clip = 50
        palpha = 0.05
        pgamma = 0.05
        pbeta = 0.05
        c_len = 10
        save_p = "Contextual_LDA_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"

        st.write("Unggah file preprocessing untuk mengekstraksi file.")
        uploaded_files1 = st.file_uploader("Pilih file txt", type=["txt"], accept_multiple_files=True)
        
        for uploaded_file in uploaded_files1:
            bytes_data = uploaded_file.read()
            bytes_data = bytes_data.decode()
  
            bytes_data = bytes_data.split(". ")
            for i in range(len(bytes_data)):
                bytes_data[i] = bytes_data[i].split(" ")                           
            st.session_state.txt_upload.append(bytes_data)

        st.success("Upload teks berhasil!")
        st.subheader("Hasil Upload Teks")

        for idx, upload_t in enumerate(st.session_state.txt_upload):
            print(f"upload {idx}")
            st.write(f"Hasil Upload Preprocessing File {idx + 1}:")
            st.text_area("Output Preprocessing" ,value=upload_t, height=300)
            result = run_clda(upload_t, start, end, iteration_num, save_p, clip, c_len, palpha, pbeta, pgamma)
            if result is not None:
                st.dataframe(result[1])
                doc_lens = [len(d) for d in result[1].Text]

                    # Plot
                plt.figure(figsize=(8,5), dpi=160)
                plt.hist(doc_lens, bins = 35, color='navy')
                plt.text(120, 4, "Mean   : " + str(round(np.mean(doc_lens))))
                plt.text(120, 3.5, "Median : " + str(round(np.median(doc_lens))))
                plt.text(120, 3, "Stdev   : " + str(round(np.std(doc_lens))))
                plt.text(120, 2.5, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
                plt.text(120, 2, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

                plt.gca().set(xlim=(0, 100), ylabel='Number of Documents', xlabel='Document Word Count')
                plt.tick_params(size=16)
                plt.xticks(np.linspace(0,150,9))
                plt.title('Distribution of Document Word Counts', fontdict=dict(size=20))
                    # plt.show()
                st.pyplot(plt.gcf())

                from wordcloud import WordCloud
                # define subplot grid
                fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(15, 12))
                plt.subplots_adjust(hspace=0.5)
                fig.suptitle("Word Cloud Dominant Topic Words", fontsize=18, y=0.95)
                
                p_df = result[1].copy()
                new_ = p_df.groupby(['dominant_topic'], as_index = False).agg({'sentence': ' '.join})

                # loop through tickers and axes
                for item, i, ax in zip(new_.sentence, new_.dominant_topic, axs.ravel()):
                    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(item)
                    ax.imshow(word_cloud, interpolation='bilinear')
                    ax.set_title(f"Topic - {i}".upper())
                    ax.set_xlabel("")
                st.pyplot(plt.gcf())
                # st.session_state.stage2 = 0 
            else:
                st.error("Error")


    if menu2 == "Ringkasan":

        if st.session_state.stage1 == 1:
            st.session_state.text4_ = st.session_state.text4
            st.session_state.merged_text2_ = st.session_state.merged_text2
        
        if st.session_state.stage2 == 1:
            st.session_state.text4_ = st.session_state.txt_upload
            text3 = []
            for item in st.session_state.text4_:
                text3.append(" ".join(item))
            text3 = " ".join(text3)
            st.session_state.merged_text2_upload.append(text3)
            st.session_state.merged_text2_ = st.session_state.merged_text2_upload

        start = 5
        end = 1
        iteration_num = 5
        clip = 50
        palpha = 0.05
        pgamma = 0.05
        pbeta = 0.05
        c_len = 10
        save_p = "Contextual_LDA_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"
        # Tampilkan pilihan untuk meringkas teks: BERT, TF-IDF, atau Hybrid-TF_IDF
        st.subheader("Pilih metode meringkas teks:")

        summary_option = st.radio("", ["BERT", "TF-IDF", "Hybrid-TF-IDF", "CLDA"])
        button_proses_ringkasan = st.button("Proses") 

        if button_proses_ringkasan:
            st.session_state.proses_ringkas = 1
            st.success("Proses Peringkasan Teks berhasil!")
        
        if summary_option == "CLDA" and st.session_state.proses_ringkas == 1:
            print("CLDA")
            st.session_state.hasil_ringkas = []
            for idx, textt in enumerate(st.session_state.text4_):
                result = run_clda(textt, start, end, iteration_num, save_p, clip, c_len, palpha, pbeta, pgamma)
                if result is not None:
                    st.session_state.hasil_ringkas.append(". ".join(result[0]))
                    st.text_area(f"Ringkasan CLDA {idx}" ,value=". ".join(result[0]), height=300)
            st.session_state.proses_ringkas = 0
            st.session_state.hasil_rouge_metric = []
        if summary_option == "BERT" and st.session_state.proses_ringkas == 1:
            print("BERT")
            st.session_state.hasil_ringkas = []
            for idx, textt in enumerate(st.session_state.merged_text2_):
                result = run_bert(textt)
                if result is not None:
                    st.session_state.hasil_ringkas.append(result)
                    st.text_area(f"Ringkasan BERT {idx}" ,value=result, height=300)
            st.session_state.proses_ringkas = 0
            st.session_state.hasil_rouge_metric = []
        if summary_option == "TF-IDF" and st.session_state.proses_ringkas == 1:
            print("TF-IDF")
            st.session_state.hasil_ringkas = []
            for idx, textt in enumerate(st.session_state.merged_text2_):
                result = run_tf_idf(textt)
                if result is not None:
                    st.session_state.hasil_ringkas.append(result)
                    st.text_area(f"Ringkasan TF IDF {idx}" ,value=result, height=300)
            st.session_state.proses_ringkas = 0
            st.session_state.hasil_rouge_metric = []
        elif summary_option == "Hybrid-TF-IDF" and st.session_state.proses_ringkas == 1:
            print("Hybrid-TF-IDF")
            st.session_state.hasil_ringkas = []
            for idx, textt in enumerate(st.session_state.merged_text2_):
                result = run_hybrid_tf_idf(textt)
                if result is not None:
                    st.session_state.hasil_ringkas.append(result)
                    st.text_area(f"Ringkasan Hybrid TF IDF {idx}" ,value=result, height=300)
            st.session_state.proses_ringkas = 0
            st.session_state.hasil_rouge_metric = []


    if menu3 == "ROUGE Metric" and st.session_state.proses_ringkas == 0:
        print("ROUGE Metric")
        st.write("Unggah file referensi.")

        uploaded_files1 = st.file_uploader("Pilih file txt", type=["txt"], accept_multiple_files=True)
        
        for uploaded_file in uploaded_files1:
            file_referensi = uploaded_file.read()
            file_referensi = file_referensi.decode()
            st.session_state.file_ref.append(file_referensi)
            st.session_state.file_name_rouge.append(uploaded_file.name)

        but_r = st.button("Tinjau Rouge Metric")

        if but_r:
            st.session_state.butt_r = 1

        if (len(st.session_state.hasil_ringkas)) > 0 and st.session_state.butt_r == 1:
            for idx, txtt in enumerate(st.session_state.hasil_ringkas):
                file_prediksi = txtt
                result_rouge_metric = run_rouge_metric(st.session_state.file_ref[idx], file_prediksi)
                st.session_state.hasil_rouge_metric.append(result_rouge_metric)
                p = pd.DataFrame(st.session_state.hasil_rouge_metric, 
                                     columns = ["Rouge 1", "Rouge 2", "Rouge L"])
                p["Name"] = ''
                for i in range(len(p)):
                    p["Name"][i] = st.session_state.file_name_rouge[i]
                p = p[["Name", "Rouge 1", "Rouge 2", "Rouge L"]]

            st.dataframe(p)
            st.session_state.hasil_ringkas = []
            st.session_state.hasil_rouge_metric = []
            st.session_state.butt_r = 0
            st.session_state.file_ref = []
            st.session_state.file_name_rouge = []

           
    if menu3 == "Perplexities":
        print("Perplexities")
        start = 5
        end = 1
        iteration_num = 5
        clip = 50
        palpha = 0.05
        pgamma = 0.05
        pbeta = 0.05
        c_len = 10
        save_p = "Contextual_LDA_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"
        
        ress = []
        if len(st.session_state.text4) > 0:
            ress = st.session_state.text4
        if len(st.session_state.txt_upload) > 0:
            ress = st.session_state.txt_upload

        for idx, textt in enumerate(ress):
            st.write(f"Perplexities File {idx + 1}:")
            dataset = save_p
            run_clda(textt, start, end, iteration_num, save_p, clip, c_len, palpha, pbeta, pgamma)
            y1 = np.load(str(dataset) +"C-LDAper_list"+ str(start) +".npy",allow_pickle=True)
            # print(y1)
            x = np.linspace(0, iteration_num+1, iteration_num+1)
            plt.figure(figsize=(8,5), dpi=160)
            plt.plot(x[::1], y1[0:], "r*-", label='C-LDA', linewidth=1)
            plt.title("Convergence Test By Perplexities")
            plt.ylabel(u"Perplexities")
            plt.xlabel(u"Iterations")
            plt.legend(loc="upper right")
            st.pyplot(plt.gcf())

if __name__ == "__main__":
    main()