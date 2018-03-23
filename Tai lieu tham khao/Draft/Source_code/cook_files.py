# -*- coding: utf-8 -*-
# @author: JapHien

import os, re, time

start_time = time.time()
folder_path = 'DATA/MATERIAL/'
save_path = 'DATA/COOKED/'

def get_stopwords():
	tmp_stop_words = []
	stop_words = []
	for file in os.listdir('./stopword/'):
		print ('Found stop word file', file)
		with open ('./stopword/' + file) as f:
			for line in f:
				line = line.strip().replace('_', ' ')
				if line not in tmp_stop_words:
					tmp_stop_words.append(line)
		for i in reversed(tmp_stop_words):
			stop_words.append(i)
	return stop_words

def cooking(file, stop_words):
    ls = []
    with open(file) as f:
        for string in f:
            if string != '':
                tmp = re.sub('[0-9;:,\.\?!%&\*\$\>\<\(\)\'\'\“\”\"\"/…\-\+]', '', string)
                tmp = re.sub('[\-\–\+\=\≥\\\]', '', tmp).strip().lower()
                tmp = re.sub('\s+', ' ', tmp)
                if tmp != '':
                    ls.append(tmp.strip())
        meal = ' '.join(ls)
        # meaningful_words = [w for w in meal.split() if not w in stop_words]
        # meal = ' '.join(meaningful_words)
    return (meal)

def cook_files(material, table):
	stop_words = get_stopwords()
	for stuff in os.listdir(material):
		print ('Working on', stuff)
		disk = open(table + re.sub('\.done', '', stuff) + '.txt', 'w')
		for item in os.listdir(material + stuff):
			content = ''
			raw =  material + stuff + os.sep + item
			meal = cooking(raw, stop_words)
			disk.write('\n'+meal)
		disk.close()

if __name__ == '__main__':
	print ('Hello World!!!')
	cook_files(folder_path,save_path)
