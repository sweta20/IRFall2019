"""
Author: Sweta Agrawal
Code to train a character seq2seq model for lemmatization
Dataset: https://github.com/michmech/lemmatization-lists
Code Reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
from __future__ import unicode_literals, print_function, division

import sys
import unicodedata
import string
import torch
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from io import open
import unicodedata
import re
import random
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

SOS_token = 0
EOS_token = 1
all_letters = string.ascii_letters + " .,;-'"
hidden_size = 100
n_letters = len(all_letters)+2
MAX_LENGTH=30

def indexesFromWord(word, data_type):
	if data_type == "tgt":
		return[all_letters.find(letter) + 2 for letter in word] + [EOS_token]
	return [all_letters.find(letter) + 2 for letter in word]

def tensorFromWord(word, data_type):
	indexes = indexesFromWord(word, data_type)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
	input_tensor = tensorFromWord(pair[1], "src")
	target_tensor = tensorFromWord(pair[0], "tgt")
	return (input_tensor, target_tensor)

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attn_weights = F.softmax(
			self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0),
								 encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()
	# this locator puts ticks at regular intervals
	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)
	plt.show()

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, decoder_type="simple"):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	loss = 0

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(
			input_tensor[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device=device)

	decoder_hidden = encoder_hidden

	for di in range(target_length):
		if decoder_type == "simple":
			decoder_output, decoder_hidden = decoder(
				decoder_input, decoder_hidden)
		else:
			decoder_output, decoder_hidden, decoder_attention = decoder(
				decoder_input, decoder_hidden, encoder_outputs)
		topv, topi = decoder_output.topk(1)
		decoder_input = topi.squeeze().detach()  # detach from history as input

		loss += criterion(decoder_output, target_tensor[di])
		if decoder_input.item() == EOS_token:
			break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length

def evaluate(encoder, decoder, word, max_length=MAX_LENGTH, decoder_type="simple"):
	with torch.no_grad():
		input_tensor = tensorFromWord(word, "src")
		input_length = input_tensor.size()[0]
		encoder_hidden = encoder.initHidden()

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[ei],
													 encoder_hidden)
			encoder_outputs[ei] += encoder_output[0, 0]

		decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

		decoder_hidden = encoder_hidden

		decoded_words = []
		for di in range(max_length):
			if decoder_type == "simple":
				decoder_output, decoder_hidden = decoder(
					decoder_input, decoder_hidden)
			else:
				decoder_output, decoder_hidden, decoder_attention = decoder(
					decoder_input, decoder_hidden, encoder_outputs)
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == EOS_token:
				# decoded_words.append('<EOS>')
				break
			else:
				# print(topi.item())
				decoded_words.append(all_letters[topi.item()-2])

			decoder_input = topi.squeeze().detach()

		return decoded_words

def load(n_letters, hidden_size, decoder_type):
	encoder = EncoderRNN(n_letters, hidden_size).to(device) 
	encoder.load_state_dict(torch.load(
		os.path.join('encoder.pt'), map_location=lambda storage, loc: storage
	).state_dict())

	if decoder_type =="simple":
		decoder = DecoderRNN(hidden_size, n_letters).to(device)
	else:
		decoder = AttnDecoderRNN(hidden_size, n_letters).to(device)
	decoder.load_state_dict(torch.load(
		os.path.join('decoder.pt'), map_location=lambda storage, loc: storage
	).state_dict())

	return encoder, decoder


def trainIters(pairs, encoder, decoder, n_epochs, print_every=1000, plot_every=100, learning_rate=0.01, decoder_type="simple"):
	plot_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
	n_iters = len(pairs)
	
	criterion = nn.NLLLoss()

	for epoch in range(n_epochs):
		training_pairs = [tensorsFromPair(random.choice(pairs))
					  for i in range(n_iters)]
		for iter in range(1, n_iters + 1):
			training_pair = training_pairs[iter - 1]
			input_tensor = training_pair[0]
			target_tensor = training_pair[1]

			loss = train(input_tensor, target_tensor, encoder,
						 decoder, encoder_optimizer, decoder_optimizer, criterion, decoder_type=decoder_type)
			print_loss_total += loss
			plot_loss_total += loss

			if iter % print_every == 0:
				print_loss_avg = print_loss_total / print_every
				print_loss_total = 0
				print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))
				# evaluateRandomly(pairs, encoder, decoder, decoder_type=decoder_type)

			if iter % plot_every == 0:
				plot_loss_avg = plot_loss_total / plot_every
				plot_losses.append(plot_loss_avg)
				plot_loss_total = 0
	showPlot(plot_losses)

def get_data(file_name):
	data = []
	with open(file_name) as f:
		for line in f:
			data.append(line.strip().split("\t"))
	return data

def evaluateRandomly(pairs, encoder, decoder, n=10, decoder_type="simple"):
	for i in range(n):
		pair = random.choice(pairs)
		print('>', pair[0])
		print('=', pair[1])
		output_words = evaluate(encoder, decoder, pair[1], decoder_type=decoder_type)
		output_sentence = ''.join(output_words)
		print('<', output_sentence)
		print('')

def main():
	data = get_data(sys.argv[1])
	mode=sys.argv[2]

	if len(sys.argv) == 4:
		decoder_type = sys.argv[3]
	else:
		decoder_type = "simple"

	if mode =="train":
		encoder = EncoderRNN(n_letters, hidden_size).to(device)

		if decoder_type =="simple":
			decoder = DecoderRNN(hidden_size, n_letters).to(device)
		else:
			decoder = AttnDecoderRNN(hidden_size, n_letters, dropout_p=0.1).to(device)
		trainIters(data, encoder, decoder, 5, print_every=1000, decoder_type=decoder_type)

		torch.save(encoder, 'encoder.pt')
		torch.save(decoder, 'decoder.pt')

	elif mode == "eval":
		encoder, decoder = load(n_letters, hidden_size, decoder_type=decoder_type)
		evaluateRandomly(data, encoder, decoder, decoder_type=decoder_type)

if __name__ == '__main__':
	main()