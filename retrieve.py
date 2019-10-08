import argparse
import pickle
from porter_stemmer import PorterStemmer
import time
from train_lemma import *
from collections import Counter
from numpy.linalg import norm

parser = argparse.ArgumentParser(description='Create an index from a collection of plain text documents')
parser.add_argument('-i', required=True, help='path to index directory')
parser.add_argument('-q', required=True, help='query')

def load_index(dir_name):
	inv_index = pickle.load( open(dir_name + "/invindex","rb"))
	params = pickle.load( open(dir_name + "/params_id","rb"))
	return inv_index, params["df"], params["preprocess"], params["N"]

def preprocess_query(query, preprocess=None):
	encoder, decoder = load(n_letters, hidden_size, decoder_type="simple")
	stemmer = PorterStemmer()
	tokens = query.split(" ")
	if preprocess == "lemma":
		lemmatized_tokens = []
		for token in tokens:
			if len(token) > 2:
				lemmatized_tokens.append(''.join(lemmatize(encoder, decoder, token.lower())))
			else:
				lemmatized_tokens.append(token.lower())
		return lemmatized_tokens
	elif preprocess == "stem":
		stemmed_tokens = []
		for token in tokens:
			if len(token) > 2:
				stemmed_tokens.append(token[:len(stemmer.stem(token.lower()))])
			else:
				stemmed_tokens.append(token)
		return stemmed_tokens
	else:
		print("Incorrect preprocess type")
		return tokens

def cosine_similarity(a, b):
	return np.dot(a, b)/ (norm(a)*norm(b))


def main():
	args = parser.parse_args()
	index_dir = args.i
	query = args.q
	k = 10

	print("[INFO]: Loading index from directory: " + index_dir)
	inv_index, df, preprocess, N = load_index(index_dir)

	start_time = time.time()

	print("[INFO]: Preprocessing data using " + preprocess  + " operation..")
	processed_query = preprocess_query(query, preprocess)
	print("[INFO: Processed query: " +  (" ").join(processed_query) )

	print("[INFO: Creating query and document vectors")
	query_word_counts = Counter(processed_query)
	query_word_counts_total = sum(query_word_counts.values())
	unique_tokens = np.unique(processed_query)

	query_vec = np.zeros(len(unique_tokens))
	doc_vectors = {}

	for i in range(len(unique_tokens)) :
		n_docs = df[unique_tokens[i]] if unique_tokens[i] in df else 0
		idf = np.log(N/(n_docs+1))
		query_vec[i] = (float(query_word_counts[unique_tokens[i]]) / query_word_counts_total) * idf

		token = unique_tokens[i]
		if token in df:
			docs = inv_index[token]

			for doc_id, tf in docs.items():
				if doc_id not in doc_vectors:
					doc_vectors[doc_id] = np.zeros(len(unique_tokens))
					doc_vectors[doc_id][i] = tf * idf
				else:
					doc_vectors[doc_id][i] = tf * idf

	print("[INFO: Computing cosine similarity between query vector and document vectors")
	list_docs = []
	for doc in doc_vectors:
		list_docs.append((doc, cosine_similarity(query_vec, doc_vectors[doc]) ))
	
	print("[INFO]: Top " + str(min(k, len(list_docs)))  + " results..")
	i = 0
	for (doc_id, score) in sorted(list_docs, key=lambda item: item[1], reverse=True):
		print(i, doc_id, score)
		i+=1
		if i == k:
			break
	print("[INFO]: Process took  " +  "{0:.5f}".format(time.time() - start_time) + " seconds..")
	

if __name__ == '__main__':
	main()