#
# wiki-rfa. G(8740, 78568). 1603 (0.1834) nodes with in-deg > avg deg (18.0), 598 (0.0684) with >2*avg.deg (Wed Jan 17 14:45:42 2018)
#

set title "wiki-rfa. G(8740, 78568). 1603 (0.1834) nodes with in-deg > avg deg (18.0), 598 (0.0684) with >2*avg.deg"
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
set output 'inDeg.wikiPlot.png'
plot 	"inDeg.wikiPlot.tab" using 1:2 title "" with linespoints pt 6
