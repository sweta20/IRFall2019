import pandas as pd
import argparse
from train_lemma import *
from collections import Counter
from itertools import chain
import numpy as np 
import os
import pickle 
import io
from nltk.corpus import stopwords
from porter_stemmer import PorterStemmer
from preprocess import get_paragraphs, get_sentences, get_tokens
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create an index from a collection of plain text documents')
parser.add_argument('-d', required=True, help='path to the input file')
parser.add_argument('-i', required=True, help='output index directory')


def prepare_data(filename):
	df = pd.read_csv(filename, header=0, encoding="ISO-8859-1")
	text = []
	for index, row in df.iterrows():
		text.append(str(row['_unit_id']) + "\t" + row['headline'] + " " + row['text'].replace("</br></br>","\n") + "\n")
	return text

def read_file(file_path):
	data = []
	for line in open(file_path):
		tokens = line.strip().split(' ')
		data.append(tokens)
	return data

def create_vocab(data):
	print("[INFO]: Creating Vocab: ")
	only_text = data.values()
	flatten = lambda l: [item for sublist in l for item in sublist]
	only_text = [flatten(para)  for para in only_text]
	word_freq = Counter(chain(*only_text))
	words = list(word_freq.keys())
	print("Vocabulary created of size: " + str(len(words)))
	id2word = {i : words[i] for i in range(len(words)) }
	return id2word

def create_invertedindex_matrix(data):
    id2word = create_vocab(data)
    
    word2id = {v: k for k, v in id2word.items()}
    
    inv_index = { i : {} for i in id2word.keys() }
    for key in data.keys():
        doc = data[key]
        for para in doc:
            for word in para:
                if key not in inv_index[word2id[word]]:
                    inv_index[word2id[word]][key] =1
                else:
                    inv_index[word2id[word]][key] +=1
    for key in inv_index.keys():
        inv_index[key] = sorted(inv_index[key].items(), key=lambda item: item[1], reverse=True)

    return inv_index, word2id


def preprocess_input(data, preprocess="stem", out_file=None):
	encoder, decoder = load(n_letters, hidden_size, decoder_type="simple")
	all_preprocessed = {}
	for line in tqdm(data):
		unit_id = line.split("\t")[0]
		text = ("\t").join(line.split("\t")[1:])
		if out_file is not None:
			out_file.write(unit_id + "\t")
		paragraphs = get_paragraphs(text)
		stemmer = PorterStemmer()
		preprocessed = []
		for paragraph in paragraphs[:-1]:
			sentences = get_sentences(paragraph)
			preprocessed_sent = [] 
			for sentence in sentences:
				tokens = get_tokens(sentence)
				if preprocess == "lemma":
					lemmatized_tokens = []
					for token in tokens:
						if len(token) > 2:
							lemmatized_tokens.append(''.join(lemmatize(encoder, decoder, token.lower())))
						else:
							lemmatized_tokens.append(token.lower())
					if out_file is not None:
						out_file.write( (" ").join(lemmatized_tokens) )
					preprocessed_sent.append(lemmatized_tokens)
				elif preprocess == "stem":
					stemmed_tokens = []
					for token in tokens:
						if len(token) > 2:
							stemmed_tokens.append(token[:len(stemmer.stem(token.lower()))])
						else:
							stemmed_tokens.append(token)
					preprocessed_sent.extend(stemmed_tokens)
					if out_file is not None:
						out_file.write( (" ").join(stemmed_tokens))
				if out_file is not None:
					out_file.write(" ")
			preprocessed.append(preprocessed_sent)
			if out_file is not None:
				out_file.write("\n")
		all_preprocessed[unit_id] = preprocessed

	return all_preprocessed

def main():
	args = parser.parse_args()
	input_file = args.d
	output_dir = args.i
	input_name = input_file.split(".")[0] 

	print("[INFO]: Extracting headlines and text from the csv file: " + input_file)
	processed=prepare_data(input_file)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	preprocess="stem"
	print("[INFO]: Preprocessing data using " + preprocess  + " operation..")
	all_preprocessed = preprocess_input(processed, preprocess)

	print("[INFO]: Creating index on preprocess data ")
	inv_index, word2id = create_invertedindex_matrix(all_preprocessed)
	print("Processed " + str(len(all_preprocessed)) + " documents.")

	print("[INFO]: Saving index to directory: " + output_dir)
	pickle.dump(inv_index, open(output_dir + "/invindex","wb"))
	pickle.dump({"vocab": word2id, "preprocess" : preprocess, "N": len(all_preprocessed)}, open(output_dir + "/params",'wb'))

if __name__ == '__main__':
	main()