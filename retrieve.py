import argparse
import pickle
from porter_stemmer import PorterStemmer
import time
from train_lemma import *
from collections import Counter
from numpy.linalg import norm
from gensim import models
from nltk.corpus import wordnet
import pandas as pd

parser = argparse.ArgumentParser(description='Create an index from a collection of plain text documents')
parser.add_argument('-i', required=True, help='path to index directory')
parser.add_argument('-q', required=True, help='query')
parser.add_argument('-e', action='store_true', help='embeddings based expansion')
parser.add_argument('-l', action='store_true', help='wordnet based expansion')

def prepare_data(filename):
	df = pd.read_csv(filename, header=0, encoding="ISO-8859-1")
	text = {}
	for index, row in df.iterrows():
		text[str(row['_unit_id'])] =  row['headline'] + " " + row['text'].replace("</br></br>","\n")
	return text

def load_index(dir_name):
	inv_index = pickle.load( open(dir_name + "/invindex","rb"))
	params = pickle.load( open(dir_name + "/params_id","rb"))
	return inv_index, params["vocab"], params["preprocess"], params["N"]

def load_w2v():
	print("[INFO]: Loading Word2Vec")
	return models.KeyedVectors.load_word2vec_format(
	'GoogleNews-vectors-negative300.bin', binary=True)

def get_k_nearest_word2vec(query_tokens, k=5):
	print("[INFO]: Expanding the query using nearest neighbors to query tokens")
	neighbors = []
	for token in query_tokens:
		results = word2vec.similar_by_word(token)
		neighbors.extend([x[0] for x in results])
	counts = Counter(neighbors)
	return [x[0] for x in counts.most_common(k*len(query_tokens))]

def get_k_nearest_wordnet(query_tokens, k=5):
	print("[INFO]: Expanding the query using wordnet")
	synonyms = []
	for token in query_tokens:
		for syn in wordnet.synsets(token):
			for lemma in syn.lemmas() :
				if lemma.name() not in synonyms:
					synonyms.append(lemma.name())
	counts = Counter(synonyms)
	print(counts)
	return [x[0] for x in counts.most_common(k*len(query_tokens))]


def preprocess_query(query, preprocess=None, expand=None):
	encoder, decoder = load(n_letters, hidden_size, decoder_type="simple")
	stemmer = PorterStemmer()
	tokens = query.split(" ")

	if expand == "e":
		 query_tokens = get_k_nearest_word2vec(tokens)
		 tokens += query_tokens

	if expand == "l":
		 query_tokens = get_k_nearest_wordnet(tokens)
		 tokens += query_tokens

	print("[INFO: query after expansion: " +  (" ").join(tokens) )

	tokens = set(tokens)

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
	dot_p = np.dot(a, b)
	norm_a = norm(a, 2)
	norm_b = norm(b, 2)
	return dot_p / (norm_a*norm_b)

def process_query(query, inv_index, words, preprocess, N , embed_expansion, k=5):
	start_time = time.time()
	print("[INFO]: Preprocessing data using " + preprocess  + " operation..")
	processed_query = preprocess_query(query, preprocess, embed_expansion)

	print("[INFO: Creating query and document vectors")
	query_word_counts = Counter(processed_query)
	query_word_counts_total = sum(query_word_counts.values())
	unique_tokens = np.unique(processed_query)

	dictOfWords = { i : words[i] for i in range(0, len(words) ) }
	invIndex = {v: k for k, v in dictOfWords.items()}

	query_vec = np.zeros(len(words))
	doc_vectors = {}

	for i in range(len(unique_tokens)) :
		token = unique_tokens[i]
		if token in words:
			docs = inv_index[token]
			n_docs = len(docs)
			idf = np.log(N/(n_docs+1))

			query_vec[invIndex[token]] = (float(query_word_counts[unique_tokens[i]]) / query_word_counts_total) * idf

			for doc_id, tf in docs.items():
				doc_vectors[doc_id] = np.zeros(len(words))
				for j in range(len(words)):
					if doc_id in inv_index[words[j]]:
						doc_vectors[doc_id][j] = inv_index[words[j]][doc_id] * idf


	print("[INFO: Computing cosine similarity between query vector and document vectors")
	list_docs = []
	for doc in doc_vectors:
		list_docs.append((doc, cosine_similarity(query_vec, doc_vectors[doc])))
	
	print("[INFO]: Top " + str(min(k, len(list_docs)))  + " results..")
	i = 0
	for (doc_id, score) in sorted(list_docs, key=lambda item: item[1], reverse=True):
		# print(i, doc_id, "{0:.3f}".format(score), all_data[doc_id], "\n\n")
		print(i, doc_id, "{0:.3f}".format(score))
		i+=1
		if i == k:
			break
	print("[INFO]: Process took  " +  "{0:.2f}".format(time.time() - start_time) + " seconds..")

def main():
	args = parser.parse_args()
	index_dir = args.i
	query = args.q
	embed_expansion_embed = args.e
	embed_expansion_ling = args.l

	print("[INFO]: Loading index from directory: " + index_dir)
	inv_index, words, preprocess, N = load_index(index_dir)

	if embed_expansion_embed:
		global word2vec
		word2vec = load_w2v()

	# global all_data
	# all_data = prepare_data("Full-Economic-News-DFE-839861.csv")

	if embed_expansion_embed:
		embed_expansion = "e"
	elif embed_expansion_ling:
		embed_expansion = "l"
	else:
		embed_expansion = "n"

	# for query in ["Health insurance", "Employee drug abuse", "Market stocks decline", "Obama republican",
	# "Business Canada", "Gasoline prices hike", "Infosys company branch China", "Ivy league university competition grades",
	# "Shopping local deals", "Global warming health"]:
	process_query(query, inv_index, words, preprocess, N, embed_expansion)
	

if __name__ == '__main__':
	main()