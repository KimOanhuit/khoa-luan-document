#
# wiki-RFA In Degree. G(11401, 189004). 1961 (0.1720) nodes with in-deg > avg deg (33.2), 1068 (0.0937) with >2*avg.deg (Tue Nov 28 09:23:47 2017)
#

set title "wiki-RFA In Degree. G(11401, 189004). 1961 (0.1720) nodes with in-deg > avg deg (33.2), 1068 (0.0937) with >2*avg.deg"
set key bottom right
set logscale xy 10
set format x "10^{%L}"
set mxtics 10
set format y "10^{%L}"
set mytics 10
set grid
set xlabel "In-degree"
set ylabel "Count"
set tics scale 2
set terminal png font arial 10 size 1000,800
set output 'inDeg.wikiInDeg.png'
plot 	"inDeg.wikiInDeg.tab" using 1:2 title "" with linespoints pt 6
