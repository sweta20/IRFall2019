"""
Regex: 

Author: Sweta Agrawal
- https://docs.python.org/3.2/library/re.html
- https://regex101.com/

Usage: python main.py <inp_file> <stem/lemma> <out_file>

"""
import re
import sys
from train_lemma import *
from porter_stemmer import PorterStemmer

import warnings
warnings.filterwarnings("ignore")

 
def read_file(file_name):
	return open(file_name, encoding="utf8", errors='ignore').read()

def get_paragraphs(text):
	return text.split("\n")

def get_sentences(para):

	"""
	Pattern: [(?<=[^A-Z].[.?!])][ ][+(?=[A-Z])]
	[(?<=[^A-Z].[.?!])]: a single character not in list followed by any character represented by . followed by line terminators .?!
	[ ]: matches " "
	[+(?=[A-Z])]: matches between one and many times a single character in [A-Z]

	The main idea is that the fullstop is usually followed by a capital letter.
	"""

	regex_pattern = r'(?<=[^A-Z].[.?!]) +(?=[A-Z])'
	return re.split(regex_pattern, para)

def get_tokens(sent):

	"""
	Pattern:
	[^\s\w]: anything not in [A-Za-z0-9_] and space -> identify punctuation
	[\w'.-]+ matches a character in the list [a-zA-Z0-9_.-'] for zero or more occurences

	The main idea is that the line terminator is already taken care by sentence splitter is usually followed by a capital letter.
	"""

	regex_pattern = r"[\w'.-]+|[^\s\w]"
	tokens = re.findall(regex_pattern, sent[:-1])

	if len(sent) > 0:
		tokens += [sent[-1]]

	return tokens

def lemmatize(encoder, decoder, token):
	
	"""
	The lemmatizer uses the learned char RNN model to predict the lematized version.
	"""

	return evaluate(encoder, decoder, token)

def preprocess_input(inp_file, preprocess, out_file=None):
	encoder, decoder = load(n_letters, hidden_size, decoder_type="simple")
	text = read_file(inp_file)

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
				preprocessed_sent.append(stemmed_tokens)
				if out_file is not None:
					out_file.write( (" ").join(stemmed_tokens))
			if out_file is not None:
				out_file.write(" ")
		preprocessed.append(preprocessed_sent)	
		if out_file is not None:
			out_file.write("\n")

	return preprocessed

def main():
	inp_file = sys.argv[1]
	preprocess = sys.argv[2] # choose lemma or stem
	out_file = open(sys.argv[3], "w")

	preprocess_input(inp_file, preprocess, out_file)

if __name__ == '__main__':
	main()