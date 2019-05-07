import numpy as np

# cosine similarity using tf-idf
def cos_similarity(query_video, doc_model):
    num = np.dot(query_video, doc_model)
    denom1 = np.sqrt(np.sum(np.square(query_video)))
    denom2 = np.sqrt(np.sum(np.square(doc_model)))
    denom = denom1 * denom2
    score =  num / denom
    return score