"""
Author: Sweta Agrawal
Implementation of Porter Stemmer
Reference:
 - http://snowball.tartarus.org/algorithms/porter/stemmer.html
 - https://tartarus.org/martin/PorterStemmer/def.txt
"""

dict_1 = {'abli': 'able', 'anci': 'ance', 'eli': 'e', 'izer': 'ize', 'ousness': 'ous', 
'ator': 'ate', 'ation': 'ate', 'enci': 'ence', 'alli': 'al', 'ational': 'ate', 
'biliti': 'ble', 'ousli': 'ous', 'fulness': 'ful', 'alism': 'al', 'iviti': 'ive',
'entli': 'ent', 'iveness': 'ive', 'aliti': 'al', 'tional': 'tion', 'ization': 'ize'}

dict_2 = {"icate":"ic","ative":"","alize":"al","iciti":"ic","ical":"ic","ful":"","ness":""}

ending_list = ["al","ance","ence","er","ic","able","ible","ant","ement","ment","ent","ou","ism",
"ate","iti","ous","ive","ize"]

class PorterStemmer():

	def is_cons(self, word, i):
		if i!=0 and self.is_cons(word, i-1) and word[i] in 'y':
			return False
		else:
			if word[i] in 'aeiou':
				return False
			else:
				return True

	def word_form(self, word):
		form = ''

		for i in range(len(word)):
			if i-1 > 0:
				if self.is_cons(word, i):
					if form[-1] != 'C':
						form += 'C' 
				else:
					if form[-1] != 'V':
						form += 'V'
			else:
				if self.is_cons(word, i):
					form += 'C'
				else:
					form += 'V'

		return form

	def measure(self, word):
		word_form = self.word_form(word)
		return word_form.count('VC')

	def step_1a(self, word):
		if word.endswith("sses"):
			word = word[:-4]
		elif word.endswith("ies"):
			word = word[:-3]
		elif word.endswith("ss"):
			word = word[:-2]
		elif word.endswith("s"):
			word = word[:-1]
		return word

	def check_doublecons(self, word):
		i = len(word)-1
		if (i-1>0)  and (word[i]==word[i-1]) and (self.is_cons(word, i)):
			return True
		return False

	def check_cvc(self, word):
		if len(word) >= 3 and self.word_form(word).endswith("CVC") and (word[-1] not in "wxy"):
			return True
		return False

	def step_1b_I(self, word):
		if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
			word += "e"
		elif self.check_doublecons(word) and not (word.endswith("l") or word.endswith("z") or word.endswith("s")):
			word = word[:-1]
		elif self.measure(word) == 1 and self.check_cvc(word):
			word+="e"

		return word

	def step_1b(self, word):
		
		m = self.measure(word)

		if word.endswith("eed"):
			if self.measure(word[:-3]) > 0:
				word = word[:-3] + "ee"
		elif word.endswith("ed"):
			if "V" in self.word_form(word[:-2]):
				word = word[:-2]
				word = self.step_1b_I(word)

		elif word.endswith("ing"):
			if "V" in self.word_form(word[:-3]):
				word = word[:-3]
				word = self.step_1b_I(word)

		return word

	def step_1c(self, word):
		if word.endswith("y"):
			if "V" in self.word_form(word[:-1]):
				word = word[-1] + "i"
		return word

	def step_2_and_3(self, word, dict):
		for key in dict.keys():
			if word.endswith(key) and self.measure(word[:-len(key)]) > 0:
				word = word[:-len(key)] + dict[key]
		return word

	def step_4(self, word):
		for suffix in ending_list:
			if(self.measure(word[:-len(suffix)]) > 1):
				if word.endswith(suffix):
					word = word[:-len(suffix)]
				elif word.endswith("ion") and word[-4] in "st":
					word = word[:-3]
		return word

	def step_5a(self, word):
		if(self.measure(word[:-1]) > 1):
			if word.endswith("e"):
				word = word[:-1]
		elif (self.measure(word[:-1]) == 1) and not self.check_cvc(word):
			if word.endswith("e"):
				word = word[:-1]
		return word

	def step_5b(self, word):
		if(self.measure(word) > 1) and self.check_doublecons(word) and word[-1] == "l":
			word = word[-1]
		return word

	def stem(self, word):
		word = self.step_1a(word)
		word = self.step_1b(word)
		word = self.step_1c(word)
		word = self.step_2_and_3(word, dict_1)
		word = self.step_2_and_3(word, dict_2)
		word = self.step_4(word)
		word = self.step_5a(word)
		word = self.step_5b(word)
		return word

