from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
import spacy

app = Flask(__name__)
nlp = spacy.load('en_core_web_md')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    output=[]
    file1 = request.files['file1']
    file2 = request.files['file2']

    text1 = file1.read().decode('utf-8')
    text2 = file2.read().decode('utf-8')
    
    if file1.filename.endswith('.py'):
        text1 = file1.read().decode('utf-8')
        
    elif file1.filename.endswith('.cpp'):
        text1 = file1.read().decode('utf-8', errors='ignore')

    if file2.filename.endswith('.py'):
        text2 = file2.read().decode('utf-8')
    elif file2.filename.endswith('.cpp'):
        text2 = file2.read().decode('utf-8', errors='ignore')
        
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix1 = tfidf_vectorizer.fit_transform([text1])
    tfidf_matrix2 = tfidf_vectorizer.transform([text2])
    
    cos_sim = cosine_similarity(tfidf_matrix1, tfidf_matrix2)[0][0]

    fuzz_ratio = fuzz.token_sort_ratio(text1, text2)
    set1 = set(text1.split())
    set2 = set(text2.split())
    common_words = list(set1 & set2)
    jaccard_sim = len(common_words) / len(set1 | set2)

    seq_matcher = SequenceMatcher(None, text1, text2)
    seq_match_sim = seq_matcher.ratio()

    doc1 = nlp(text1)
    doc2 = nlp(text2)
    word2vec_sim = doc1.similarity(doc2)

    # CBOW
    cbow_sim = 0.0
    doc2 = nlp(text2)
    for token in doc1:
        if token.has_vector:
            cbow_sim += token.similarity(doc2)
    cbow_sim /= len(doc1)

    # Doc2Vec
    doc1_vec = nlp(text1).vector
    doc2_vec = nlp(text2).vector
    doc2vec_sim = cosine_similarity([doc1_vec], [doc2_vec])[0][0]

    result = {
        'cosine_similarity_tfidf': round(float(cos_sim), 2),
        'fuzzy_match_ratio': fuzz_ratio/100, 
        'jaccard_similarity': round(float(jaccard_sim), 2),
        'sequence_match_similarity': round(float(seq_match_sim), 2),
        'word2vec_similarity': round(float(word2vec_sim), 2),
        'cbow_similarity': round(float(cbow_sim), 2),
        'doc2vec_similarity': round(float(doc2vec_sim), 2),
        'common_words': common_words
    }
    
    print(result)
    return render_template('display.html',result= result)


if __name__ == '__main__':
    app.run(debug=True)
