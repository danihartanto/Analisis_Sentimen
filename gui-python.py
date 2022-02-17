import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import re
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import streamlit as st
from st_aggrid import AgGrid
import plotly.express as px
import plotly.figure_factory as ff

pd.set_option("display.max_rows", None, "display.max_columns", None)
st.set_page_config(layout="wide")

def praproses(text):
    text = text.lower() #huruf besar jadi huruf kecil
    text = text.strip() #White spaces removal
    text = re.sub('@[\w]+','',text) #remove username tweet
    text = re.sub(r'http\S+', '', text) #remove link http
    text = re.sub('\d+', ' ', text) #remove angka
    text = re.sub(r'[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', '', text)  #remove punctuation

    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in text:
        if ele in punc:
            text = text.replace(ele, "")
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    text = emoji_pattern.sub(r'', text) # no emoji
    text = text.split() #memisahkan teks kata per kata
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text =  [stemmer.stem(word) for word in text]
    text = " ".join([word for word in text])
    return text

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
def tf_idf(data):
    vectorizer = TfidfVectorizer(max_df=1.0,min_df=1,norm=None)
    respon = vectorizer.fit_transform(data).toarray()
    return respon

def main():
    image = Image.open('himatif.png')
    col1, col2= st.sidebar.columns([4, 6])
    col1.image(image,width=100, use_column_width=None)
    col2.write("Analisis Sentimen \n Masyarakat pasca Pandemi di \n Indonesia")
    add_selectbox = st.sidebar.selectbox("Silahkan upload dataset disini!",
                                         ("Upload dataset kotor", "Upload dataset bersih"))
    st.header("Analisa sentimen masyarakat soal Kuliah Daring di Indonesia")
    if add_selectbox == "Upload dataset kotor":
        uploaded_file = st.sidebar.file_uploader("Upload data")
        st.text("Tahapan == upload data asli -> Pengolahan data -> simpan data -> klasifikasi -> uji masukan ")
        if uploaded_file is not None:
            dataset = pd.read_csv(uploaded_file)
            #st.table(dataset)
            AgGrid(dataset)
            lihat_data = st.checkbox("Lihat data karakteristik data")
            if lihat_data:
                positive = dataset['polarity'].value_counts()[1]
                negative = dataset['polarity'].value_counts()[-1]
                fig, ax = plt.subplots(figsize = (10,5))
                index = range(2)
                plt.bar(index[1], positive, color='green', edgecolor = 'black', width = 0.8)
                plt.bar(index[0], negative, color = 'orange',edgecolor = 'black', width = 0.8)
                plt.legend([[positive ,"positif"], [negative,"negatif"] ])
                plt.xlabel('Sentiment Status ',fontdict = {'size' : 15})
                plt.ylabel('Sentimental Frequency', fontdict = {'size' : 15})
                plt.title("Presentase jumlah sentimen", fontsize = 20)
                st.pyplot(fig)
                                
            add_checkbox = st.checkbox("Pengolahan data")
            if add_checkbox:
                dataset['text_clean'] = (dataset['tweet'].astype(str)).apply(lambda x: praproses(x))
                datas = dataset['text_clean']
                X = tf_idf(datas)
                y = dataset['polarity']
                
                st.write("hasil pengolahan data")
                
                hasil_olah_data = (dataset[["tweet","text_clean","polarity"]])
                AgGrid(hasil_olah_data)
                save_checkbox = st.checkbox("Save file ke csv")
                
                if save_checkbox:
                    save = dataset.to_csv("data-bersih.csv",index=False)
                    print(save)
                    st.caption('Berhasil disimpan')
                st.subheader("Tahapan klasifikasi")
                folds = st.slider("Geser slider ke kanan untuk menentukan jumlah fold!", min_value=1, max_value=10, value=1)
                klasifikasi_checkbox = st.checkbox("Klasifikasi")
                
                if klasifikasi_checkbox:
                    kf = KFold(n_splits=folds, random_state=None)
                    iterasi = 0
                    akurasi_rata=list()
                    recall_rata=list()
                    presisi_rata=list()
                    f_score_rata=list()
                    for train_index, test_index in kf.split(X):
                        #print("Train:", train_index, "\nTest:",test_index,'\n')
                        iterasi+=1
                        X_train, X_test = X[train_index], X[test_index] 
                        y_train, y_test = y[train_index], y[test_index]
                        modelSVM = SVM()
                        modelSVM.fit(X_train,y_train)
                        #modelSvm = SVM.fit(X_train,y_train)
                        y_pred = modelSVM.predict(X_test)
                            
                        akurasi = accuracy_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        presisi = precision_score(y_test, y_pred)
                        f_score = f1_score(y_test, y_pred)

                        akurasi_rata.append(akurasi)
                        recall_rata.append(recall)
                        presisi_rata.append(presisi)
                        f_score_rata.append(f_score)
                            
                        data = [{"K Fold" : ('fold ke-',iterasi), "Akurasi": akurasi,"Recall":recall,"Presisi":presisi,"F1 Score":f_score}]
                        df = pd.DataFrame(data=data)
                            #AgGrid(df)

                    akurasi = akurasi_rata
                    recall = recall_rata
                    presisi = presisi_rata
                    fscore = f_score_rata
                    fold = 0
                    for i in range(len(akurasi)):
                        #indeks = len(akurasi)
                        result_akurasi = (akurasi[fold])*100
                        result_recall = (recall[fold])*100
                        result_presisi = (presisi[fold])*100
                        result_fscore = (fscore[fold])*100
                        st.write("Nilai fold %d dengan Akurasi %.2f , Recall= %.2f , Presisi= %.2f , F1 Score= %.2f "%(fold+1,result_akurasi,result_recall,result_presisi,result_fscore))
                        fold+=1

                    rata_acc = (sum(akurasi_rata) / len(akurasi_rata))*100
                    rata_rec = (sum(recall_rata) / len(recall_rata))*100
                    rata_pre = (sum(presisi_rata) / len(presisi_rata))*100
                    rata_fscore = (sum(fscore) / len(fscore))*100
                    st.write("")
                    st.write("Rata-rata hasil uji dengan %d fold"%(folds))
                    st.write("Akurasi= %.2f , Recall= %.2f , Presisi= %.2f , F1 Score= %.2f"%(rata_acc,rata_rec,rata_pre,rata_fscore))

                    tombol_input = st.checkbox("Tes masukan")
                    if tombol_input:
                        data = st.text_input('Masukkan sentimen/kalimat dalam bahasa indonesia')
                        #st.write('kalimat masukan: ', data)
                        tfidf = TfidfVectorizer()
                        tfidf.fit(dataset['text_clean'])
                        #text = "udah oktober april putus ngeluh"
                        output = tfidf.transform([data])
                        if modelSVM.predict(output) == 1:
                            st.write(data,"======== adalah positif - sentimen")
                        elif modelSVM.predict(output) == -1:
                            st.write(data,"======== adalah negatif - sentimen")
                

    elif add_selectbox == "Upload dataset bersih":
        uploaded_bersih = st.sidebar.file_uploader("Upload dataset setelah di olah")
        if uploaded_bersih is not None:
            dataset = pd.read_csv(uploaded_bersih)
            AgGrid(dataset[['text_clean','polarity']])

            datas = dataset['text_clean']
            X = tf_idf(datas)
            y = dataset['polarity']

            st.subheader("Tahapan klasifikasi")
            folds = st.slider("Geser slider ke kanan untuk jumlah fold!", min_value=1, max_value=10, value=1)
            klasifikasi_checkbox = st.checkbox("Klasifikasi")
            if klasifikasi_checkbox:
                
                
                kf = KFold(n_splits=folds, random_state=None)
                iterasi = 0
                akurasi_rata=list()
                recall_rata=list()
                presisi_rata=list()
                f_score_rata=list()
                for train_index, test_index in kf.split(X):
                    #print("Train:", train_index, "\nTest:",test_index,'\n')
                    iterasi+=1
                    X_train, X_test = X[train_index], X[test_index] 
                    y_train, y_test = y[train_index], y[test_index]
                    modelSVM = SVM()
                    modelSVM.fit(X_train,y_train)
                    #modelSvm = SVM.fit(X_train,y_train)
                    y_pred = modelSVM.predict(X_test)
                        
                    akurasi = accuracy_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    presisi = precision_score(y_test, y_pred)
                    f_score = f1_score(y_test, y_pred)

                    akurasi_rata.append(akurasi)
                    recall_rata.append(recall)
                    presisi_rata.append(presisi)
                    f_score_rata.append(f_score)
                        
                    data = [{"K Fold" : ('fold ke-',iterasi), "Akurasi": akurasi,"Recall":recall,"Presisi":presisi,"F1 Score":f_score}]
                    df = pd.DataFrame(data=data)
                        #AgGrid(df)

                akurasi = akurasi_rata
                recall = recall_rata
                presisi = presisi_rata
                fscore = f_score_rata
                fold = 0
                for i in range(len(akurasi)):
                    #indeks = len(akurasi)
                    result_akurasi = (akurasi[fold])*100
                    result_recall = (recall[fold])*100
                    result_presisi = (presisi[fold])*100
                    result_fscore = (fscore[fold])*100
                    st.write("Nilai fold %d dengan Akurasi %.2f , Recall= %.2f , Presisi= %.2f , F1 Score= %.2f "%(fold+1,result_akurasi,result_recall,result_presisi,result_fscore))
                    fold+=1

                rata_acc = (sum(akurasi_rata) / len(akurasi_rata))*100
                rata_rec = (sum(recall_rata) / len(recall_rata))*100
                rata_pre = (sum(presisi_rata) / len(presisi_rata))*100
                rata_fscore = (sum(fscore) / len(fscore))*100
                st.write("")
                st.write("Rata-rata hasil uji dengan %d fold"%(folds))
                st.write("Akurasi= %.2f , Recall= %.2f , Presisi= %.2f , F1 Score= %.2f"%(rata_acc,rata_rec,rata_pre,rata_fscore))



                tombol_input = st.checkbox("Tes masukan")
                if tombol_input:
                    data = st.text_input('Masukkan sentimen/kalimat dalam bahasa indonesia')
                    #st.write('kalimat masukan: ', data)
                    tfidf = TfidfVectorizer()
                    tfidf.fit(dataset['text_clean'])
                    #text = "udah oktober april putus ngeluh"
                    output = tfidf.transform([data])
                    if modelSVM.predict(output) == 1:
                        st.write(data,"======== adalah positif - sentimen")
                    elif modelSVM.predict(output) == -1:
                        st.write(data,"======== adalah negatif - sentimen")




if __name__ == "__main__":
    main()
