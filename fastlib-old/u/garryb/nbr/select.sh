
#FIELDS="/results/n_points /results/n_exemplars /results/netsim /results/n_iterations /iter_times_total/results/avg /iter_times_total/results/min /iter_times_total/results/med /iter_times_total/results/max"
FIELDS="/results/n_points /results/n_exemplars /results/netsim /results/n_iterations /iter_times_total/results/avg /iter_times_total/results/sum"
APFIELDS="/results/n_points /results/n_exemplars /results/netsim /results/n_iterations /results/avg"

E=fx-csv
$E a-lcdm-1 ./affinity $FIELDS > r-lcdm.csv
$E a-lcdm-1 ./apcluster $APFIELDS > r-ap.csv
$E a-naive-1 ./affinity $FIELDS > r-naive.csv
$E a-vis-1 ./affinity $FIELDS > r-vis.csv
