
import os
import matplotlib.pyplot as plt
import pandas as pd
from general_function import count_file_in_folder
import matplotlib

matplotlib.rc('font', size=5)

source_folder = 'DATA/MATERIAL/'
	
counts = []

for folder in os.listdir(source_folder):
	sumx = count_file_in_folder(source_folder + folder)
	counts.append((folder, sumx))

df_stats = pd.DataFrame(counts, columns=['Topic', 'Nums-of-sample'])

df_stats.plot(x='Topic', y='Nums-of-sample', kind='bar', legend=False, grid=True)
plt.show()




