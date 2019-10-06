import argparse
import pickle
from porter_stemmer import PorterStemmer
import time
from train_lemma import *

parser = argparse.ArgumentParser(description='Create an index from a collection of plain text documents')
parser.add_argument('-i', required=True, help='path to index directory')
parser.add_argument('-q', required=True, help='query')

def load_index(dir_name):
	inv_index = pickle.load( open(dir_name + "/invindex","rb"))
	params = pickle.load( open(dir_name + "/params","rb"))
	return inv_index, params["vocab"], params["preprocess"], params["N"]

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


def main():
	args = parser.parse_args()
	index_dir = args.i
	query = args.q
	score_type = "count"
	k = 10

	print("[INFO]: Loading index from directory: " + index_dir)
	inv_index, word2id, preprocess, N = load_index(index_dir)

	start_time = time.time()

	print("[INFO]: Preprocessing data using " + preprocess  + " operation..")
	processed_query = preprocess_query(query)
	print("[INFO: Processed query: " +  (" ").join(processed_query) )

	print("[INFO]: Retrieving documents using " + score_type  + " scores..")
	list_docs = []
	for token in processed_query:
		if token in word2id:
			docs = inv_index[word2id[token]]
			term_freq_total = sum([t[1] for t in docs])
			n_docs = len(docs)
			for (doc_id, count) in docs:
				if score_type == "count":
					score = count
				elif score_type == "tf":
					score = float(count)/term_freq_total
				elif score_type == "tfidf":
					score = (float(count)/term_freq_total) * (float(N)/n_docs)
				list_docs.append((doc_id, score))

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