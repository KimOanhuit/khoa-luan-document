# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import os, re
import requests
from general_function import *
from random import randrange

s1 = u'àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
s0 = u'aaaaeeeiioooouuyadiuouaaaaaaaaaaaaeeeeeeeeiioooooooooooouuuuuuuyyyy'

save_path = 'DATA/MATERIAL/'
source_url = 'https://vnexpress.net'
deep_scan_link = ['giadinh', 'suckhoe', 'thethao', 'kinhdoanh']

topic_name = {}

# remove accents
def convert(input_str):
	s = ''
	input_str = input_str.lower()
	for c in input_str:
		if c in s1:
			s += s0[s1.index(c)]
		else:
			s += c
	return s.upper()

def deep_scaner(url):
	req = requests.get(url)
	soup = BeautifulSoup(req.text, 'lxml')

	for sub in soup.find_all('nav', 'clearfix'):
		for topic in sub.find_all('a'):
			tlink = topic['href']
			if 'startup' in tlink or 'video' in tlink or 'doanh-nghiep' in tlink or 'tien-cua-toi' in tlink or 'hau-truong' in tlink or 'photo' in tlink or 'tuong-thuat' in tlink or 'cac-mon-khac' in tlink or 'du-lieu-bong-da' in tlink or 'to-am' in tlink or 'ung-thu' in tlink or 'cac-benh' in tlink:
				continue
			tname = topic.text
			link = url.strip('/') + tlink
			topic_name[link] = str(len(topic_name)) + '.' + convert(tname).strip().replace(' ', '_')

def Topic_selector(source_url):
	flag = True
	req = requests.get(source_url)
	soup = BeautifulSoup(req.text, 'lxml')

	for sub in soup.find_all('nav', 'p_menu'):
		for topic in sub.find_all('a'):
			flag = True
			tlink = topic['href'].strip()
			if tlink == '/' or tlink.startswith('//') or 'thoi-su' in tlink or 'goc-nhin' in tlink or 'tam-su' in tlink or 'cuoi' in tlink or 'cong-dong' in tlink:
				continue
			tname = topic['title']

			for sub_topic in deep_scan_link:
				if sub_topic in tlink:
					deep_scaner(tlink)
					flag = False
					break
			if flag:
				if tlink.startswith('https'):
					link = tlink
				else:
					link = source_url + tlink
				topic_name[link] = str(len(topic_name)) + '.' + convert(tname).strip().replace(' ', '_')
	for i in topic_name:
		print (i + '\t' + topic_name[i])


def Create_folder(topic_name):
	for topic in topic_name:
		if not os.path.isdir(save_path + topic_name[topic]):
			os.mkdir(save_path + topic_name[topic])

def Get_content(link):
	string =''
	req = requests.get(link)
	soup = BeautifulSoup(req.content, 'html.parser')

	# get title
	for sub in soup.find_all('h1',{'title_news_detail mb10'}):
		try:
			string = sub.string.strip() + '\n'
		except:
			string = sub.text.strip() + '\n'

	# get description
	for sub in soup.find_all('h2',{'description'}):
		try:
			string = string + sub.string.strip() + '\n'
			break
		except:
			pass
		try:
			string = string + str(sub.string).strip() + '\n'
			break
		except:
			string = ''
			return string

	# get content
	slice_place = 0
	for sub in soup.find_all('article', {'class': 'content_detail fck_detail width_common block_ads_connect'}):
		for s in sub.find_all('p'):
			# cut writer
			if '<strong><em><a title' in str(s) or '>>Thêm' in str(s):
				continue
			if ('text-align:right;' in str(s) or 'align="right"' in str(s)) and slice_place > 3:
				break
			string += ' ' + s.text.strip()
			slice_place += 1
	return string

count = 0
link = ''
num_per_topic = 200
out_of_stock = 0


# get topic links
Topic_selector(source_url)
# create storage folders# 
Create_folder(topic_name)
print (topic_name)

#
for topic in topic_name:
	init_url = topic
	print ('\n*** WORKING ON TOPIC', topic_name[topic])
	if count_file_in_folder(save_path + topic_name[topic]) > 50:
		print ('Topic was done before!\nSkip to next topic!')
		continue
	i = 0
	num_per_topic = randrange(190, 300, randrange(15, 38, 6))
	while True:
		if out_of_stock > 12:
			print ('Skip to next topic!!!')
			out_of_stock = 0
			break
		print ('PAGE', i + 1)
		url = init_url + '/page/' + str(i + 1) + '.html'
		print ('URL: ', url)
		req = requests.get(url)
		soup = BeautifulSoup(req.text,"lxml")
		for sub in soup.find_all('section',{'sidebar_1'}):
			for meat in sub.find_all('article'):

				# ignore noise
				if 'class="ic ic-video"' in str(meat) or 'class="ic ic-photo"' in str(meat) or 'class="ic ic-comment ic-x ic-invert"' in str(meat):
					continue
				try:
					link = meat.h3.a.get('href')
				except:
					link = meat.h1.a.get('href')

				if 'projects' in link or 'hoi-dap' in link or 'tu-van'in link or 'infographics' in link:
					continue
				print ('LINK', link)
				content = Get_content(link)

				# ignore short content news
				if len(content) < 1700:
					continue

				# print ('LINK', link)
				# print (content)

				# save file
				out_of_stock = 0
				file_save = save_path + topic_name[topic] + os.sep + topic_name[topic] + '_' + str(count) + ".txt"
				save = open(file_save,'w')
				save.write(content)
				save.close()

				print ('Done ', count)
				count = count + 1

				if count == num_per_topic:
					break
		if count == num_per_topic:
			break
		i += 1
		out_of_stock += 1
		
	count = 0
