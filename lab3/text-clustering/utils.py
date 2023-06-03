import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import matplotlib as plt

"""
--------------------------------------------
---------- 1. 正则表达式函数 -----------
--------------------------------------------
"""
def print_step_result(original_data, processed_data, idx_list=None):
    """
    参数：
    ----------
    original_data: 要打印的原始数据 [type: list, numpy array, pandas series]
    processed_data: 要打印的处理过的数据 [type: list, numpy array, pandas series]
    idx_list: 要打印的索引列表（可选） [type: list]
    """
    if idx_list is None:
        idx_list = [0]

    for i, (original_text, processed_text) in enumerate(zip(original_data, processed_data)):
        if i in idx_list:
            print(f"\nOriginal text: {original_text}")
            print(f"Processed text: {processed_text}")

def re_breakline(text_list, text_sub=' '):
    """
    去除中断点和回车
    :param text_list: 要准备的文本内容的列表对象 [type: list]
    :param text_sub: 字符串或模式，用于替代重码模式 [type: string]
    :return:
    """
    return [re.sub('[\n\r]', text_sub, r) for r in text_list]


def re_hiperlinks(text_list, text_sub=' link '):
    """
    替换网站和超链接为 'link'
    """

    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, text_sub, r) for r in text_list]



def re_dates(text_list, text_sub=' data '):
    """
    替换时间信息为 'data'
    """

    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(pattern, text_sub, r) for r in text_list]



def re_money(text_list, text_sub=' dinheiro '):
    """
    替换价格为 'dinheiro'
    """

    # Applying regex
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(pattern, text_sub, r) for r in text_list]


def re_numbers(text_list, text_sub=' numero '):
    """
    替换数字为 'numero'
    """

    # Applying regex
    return [re.sub('[0-9]+', text_sub, r) for r in text_list]


# [RegEx] Padrão para encontrar a palavra "não" em seus mais diversos formatos
def re_negation(text_list, text_sub=' negação '):
    """
    替换停用词中否定词为 'negação'
    """

    # Applying regex
    return [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', text_sub, r) for r in text_list]


# [RegEx] Padrão para limpar caracteres especiais
def re_special_chars(text_list, text_sub=' '):
    """
    删除特殊字符
    """

    # Applying regex
    return [re.sub('\W', text_sub, r) for r in text_list]


# [RegEx] Padrão para limpar espaços adicionais
def re_whitespaces(text_list):
    """
    删除多余的空格
    """

    # Applying regex
    white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end


"""
--------------------------------------------
-------------- 2. 去除stopwords ------------
--------------------------------------------
"""


def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
    """
    去除止语并将文本转换为小写字母
    """

    return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]


"""
--------------------------------------------
--------------- 3. STEMMING ----------------
--------------------------------------------
"""

def stemming_process(text, stemmer=RSLPStemmer()):
    """
    提取句子主干
    :param text: list，去除stopwords后的text
    :param stemmer: stemmer类型
    :return: 提取过主干后的 text
    """

    return [stemmer.stem(c) for c in text.split()]


"""
--------------------------------------------
---------------- 4. 提取特征 ----------------
--------------------------------------------
"""

# [Vocabulary] Função para aplicação de um vetorizador para criação de vocabulário
def extract_features_from_corpus(corpus, vectorizer, df=False):
    """
    Args
    ------------
    text: text to be transformed into a document-term matrix [type: string]
    vectorizer: engine to be used in the transformation [type: object]
    """

    # Extracting features
    corpus_features = vectorizer.fit_transform(corpus).toarray()
    features_names = vectorizer.get_feature_names()

    # Transforming into a dataframe to give interpetability to the process
    df_corpus_features = None
    if df:
        df_corpus_features = pd.DataFrame(corpus_features, columns=features_names)

    return corpus_features, df_corpus_features



# [Viz] Função para retorno de DataFrame de contagem por ngram
def ngrams_count(corpus, ngram_range, n=-1, cached_stopwords=stopwords.words('portuguese')):

    """
    通过ngram返回计数DataFrame的函数
    """
    # Using CountVectorizer to build a bag of words using the given corpus
    vectorizer = CountVectorizer(stop_words=cached_stopwords, ngram_range=ngram_range).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_list = words_freq[:n]

    # Returning a DataFrame with the ngrams count
    count_df = pd.DataFrame(total_list, columns=['ngram', 'count'])
    return count_df


"""
--------------------------------------------
---------------- 5. PIPELINE  -------------
--------------------------------------------
"""

# [TEXT PREP] Classe para aplicar uma série de funções RegEx definidas em um dicionário
class ApplyRegex(BaseEstimator, TransformerMixin):

    def __init__(self, regex_transformers):
        self.regex_transformers = regex_transformers

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Applying all regex functions in the regex_transformers dictionary
        for regex_name, regex_function in self.regex_transformers.items():
            X = regex_function(X)

        return X


# [TEXT PREP] Classe para aplicar a remoção de stopwords em um corpus
class StopWordsRemoval(BaseEstimator, TransformerMixin):

    def __init__(self, text_stopwords):
        self.text_stopwords = text_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(stopwords_removal(comment, self.text_stopwords)) for comment in X]


# [TEXT PREP] Classe para aplicar o processo de stemming em um corpus
class StemmingProcess(BaseEstimator, TransformerMixin):

    def __init__(self, stemmer):
        self.stemmer = stemmer

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(stemming_process(comment, self.stemmer)) for comment in X]


# [TEXT PREP] Classe para extração de features de um corpus (vocabulário / bag of words / TF-IDF)
class TextFeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, vectorizer, train=True):
        self.vectorizer = vectorizer
        self.train = train

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.train:
            return self.vectorizer.fit_transform(X).toarray()
        else:
            return self.vectorizer.transform(X)

from IPython.display import display
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture


# Dimensionality reduction
from sklearn.manifold import TSNE

# Visualization
import plotly.express as px
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
"""
--------------------------------------------
----------------- 6. 句子分析 --------------
--------------------------------------------
"""
def transform_data(df, ev, tsne=False):

    X = PCA(ev, random_state=42).fit_transform(PowerTransformer().fit_transform(df))
    X_dense = X.toarray()
    if tsne == True:
        perplexity = int(X_dense.shape[0] ** 0.5)
        X_dense = TSNE(perplexity=perplexity, random_state=42).fit_transform(X)

    return X_dense
def fit_predict_data(X, n_clusters, est='KMeans'):
    est_dict = {
        'KMeans': KMeans(n_clusters, random_state=42),
        'GaussianMixture': GaussianMixture(n_clusters, random_state=42)}

    model = est_dict[est]
    labels = model.fit_predict(X)

    return model, labels
def plot_silhouette_analysis(df, ev, n_clusters, est='KMeans', tsne=False):
    # Apply transformations to data
    X = transform_data(df, ev, tsne)

    # fit model, predict labels, and append overall average
    # score and silhouette score for each sample
    model, labels = fit_predict_data(X, n_clusters, est)
    sil_score = silhouette_score(X, labels)
    sil_sample = silhouette_samples(X, labels)

    # Create subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))


    # Part A -- silhouette plot

    # Set appropriate limits for x-axis and y-axis
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        # Get aggregated sorted silhouette scores for samples
        ith_cluster_silhouette_values = sil_sample[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.7)

        # Label silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), color = 'black', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))

        # Compute new y_lower for next plot
        y_lower = y_upper + 10

    ax1.set_title('Cluster Silhouette Plot')
    ax1.set_xlabel('Silhouette Coefficient')
    ax1.set_ylabel('Cluster')

    # Plot vertical line for overall average silhouette score
    ax1.axvline(x=sil_score, color="red", linestyle="--")

    # Set appropriate ticks for x-axis and y-axis
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    # Part B -- feature space plot

    # Plot 1st and 2nd feature space
    colors = cm.Spectral(labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, c=colors)

    # Set `transformers`; to be used in plot suptitle adjunct text
    transformers = 'PowerTransformer -> PCA -> TSNE'

    # Illuminate cluster centers if est is 'KMeans'
    if est == 'KMeans':
        centers = model.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", s=300, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        # Set `transformers`; to be used in plot suptitle adjunct text
        transformers = 'PowerTransformer -> PCA'

    ax2.set_title('Cluster Feature Space Plot')
    ax2.set_xlabel('1st Feature')
    ax2.set_ylabel('2nd Feature')

    fig.tight_layout()

    # Add suptitle and adjunct text
    plt.suptitle((f'Silhouette Analysis for {est} Clustering'), size=17)
    fig.subplots_adjust(top=0.86)
    fig.text(0.5, 0.92,
             f'Transformers: {transformers} - PCA: {ev} - n_clusters: {n_clusters}',
             ha='center',
             size=14)

    # Show output
    plt.show()
# Defining a function to plot the sentiment of a given phrase
def sentiment_analysis(text, pipeline, vectorizer, model):
    # Applying the pipeline
    if type(text) is not list:
        text = [text]
    text_prep = pipeline.fit_transform(text)
    matrix = vectorizer.transform(text_prep)

    # Predicting sentiment
    pred = model.predict(matrix)
    proba = model.predict_proba(matrix)

    # Plotting the sentiment and its score
    fig, ax = plt.subplots(figsize=(5, 3))
    if pred[0] == 1:
        text = 'Positive'
        class_proba = 100 * round(proba[0][1], 2)
        color = 'seagreen'
    else:
        text = 'Negative'
        class_proba = 100 * round(proba[0][0], 2)
        color = 'crimson'
    ax.text(0.5, 0.5, text, fontsize=50, ha='center', color=color)
    ax.text(0.5, 0.20, str(class_proba) + '%', fontsize=14, ha='center')
    ax.axis('off')
    ax.set_title('Sentiment Analysis', fontsize=14)
    plt.show()