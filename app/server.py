import numpy as np
#import pandas as pd
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from flask import Flask, render_template, request

app = Flask(__name__)

def read_article(file):
    '''
    Parameters
    ----------
    file : file url to load text file (only accepts text file)
    Returns
    -------
    sentences : list of sentences

    '''
    f = file.read()
    sentences = sent_tokenize(str(f))
    return sentences

def sentence_similarity(sent1, sent2, stopwords = None):
    '''
    Parameters
    ----------
    sent1 & sent2 : sentences 
    stopwords : common words of language which occur in all texts
    (verbal or written)
    Returns
    -------
    similarity between two input sentences

    '''
    if stopwords is None:
        stopwords = []
        
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vec1 = [0] * len(all_words)
    vec2 = [0] * len(all_words)
    
    for w in sent1:
        if w in stopwords:
            continue
        vec1[all_words.index(w)] += 1
        
    for w in sent2:
        if w in stopwords:
            continue
        vec2[all_words.index(w)] += 1
        
    return 1 - cosine_distance(vec1, vec2)    

      
def build_similarity_matrix(sentences, stop_words):
    '''
    Parameters
    ----------
    sentences : one sentence

    Returns
    -------
    similarity_matrix : matrix containing similarities b/w
    all sentences of input file

    '''
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix

def generate_summary(file_name, top_n = 2):
    '''
    Parameters
    ----------
    file_name : usually a URL for file location
    top_n : number of sentences to be returned, dynamically calculated for large files

    Returns
    -------
    multiple sentences joined by "." which can explain whole input text in short form

    '''
    stop_words = set(stopwords.words('english'))
    summarize_text = []
    
    sentences = read_article(file_name)
    
    if len(sentences) > 10:
        top_n = len(sentences)//5
    
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse = True)
    
    for i in range(top_n):
        summarize_text.append("".join(ranked_sentence[i][1]))
        
    return ". ".join(summarize_text)

  
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/success', methods=['GET','POST'])
def success_table():
    global filename
    
    if request.method=="POST":
        file=request.files['file']
        summary = generate_summary(file)
        
        return render_template("index.html", text = summary)

if __name__ == '__main__':
    app.run(debug = False)