import re
import os
import cStringIO

file = open("BaseDataset.txt", "r+")
str = file.read()

src = re.findall('SRC:(.*?)\n',str)
tgt = re.findall('TGT:(.*?)\n',str)
vot = re.findall('VOT:(-1|0|1)\n',str)
res = re.findall('RES:(-1|1)\n',str)
year = re.findall('YEA:([0-9]{4})\n',str)
date = re.findall('DAT:(.*?)\n',str)
txt = re.findall('TXT:(.*?)\n',str)

os.system("type NULL > data.csv")

file = open("data.csv","a")
file.write("SRC,TGT,VOT,RES,YEA,DAT,TXT\n")

i = 0
for d in res:
    data = src[i] + ',' + tgt[i] + ',' + vot[i] + ',' + res[i] + ',' + year[i] + ',' + date[i].replace(",","") + ',' + txt[i].replace(",","") + "\n"
    file.write(data)
    i += 1
    # print(data)

file.close()
