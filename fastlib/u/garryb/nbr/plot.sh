
cat >r-ap.csv <<__EOF__
1000, 41, -686895.549661, 168, 0.037178, 6.245904
3000, 75, -1261252.715675, 173, 0.320985, 55.530405
10000, 140, -2516505.286513, 196, 3.533092, 692.486
__EOF__

cat >r-ap-ext.csv <<__EOF__
10000, 140, -2516505.286513, 196, 3.533092, 692.486
10000000, 140, -5, 196, 3533092, 692486000
__EOF__

gnuplot <<__EOF__

set title "Scalability for a Single Iteration"
set xlabel "Number of Points"
set logscale x
set logscale y
set xrange [1000:1000000]
set ylabel "Time per Iteration (s)"
set yrange [0.01:100000]
plot "r-ap.csv" using 1:5 with linespoints pt 6 title "Frey-Dueck", \
     "r-ap-ext.csv" using 1:5 with linespoints lt 2 pt 6 title "(extrapolated)", \
     "r-lcdm.csv" using 1:5 with linespoints lt 1 pt 8 title "Dual-Tree"

set size 1.0, 0.5
set terminal postscript portrait enhanced mono dashed lw 1 "Helvetica" 14
set output "r-speed.ps"
replot

__EOF__

gnuplot <<__EOF__

set title "Total Running Time"
set xlabel "Number of Points"
set logscale x
set logscale y
set xrange [1000:1000000]
set ylabel "Time (s)"
set yrange [0.1:10000000]
plot "r-ap.csv" using 1:6 with linespoints pt 6 title "Frey-Dueck", \
     "r-ap-ext.csv" using 1:6 with linespoints lt 2 pt 6 title "(extrapolated)", \
     "r-lcdm.csv" using 1:6 with linespoints lt 1 pt 8 title "Dual-Tree"

set size 1.0, 0.5
set terminal postscript portrait enhanced mono dashed lw 1 "Helvetica" 14
set output "r-total.ps"
replot

__EOF__
