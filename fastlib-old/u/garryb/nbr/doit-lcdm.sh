
fx-run a-lcdm-1 ./affinity "--data=/local/garryb/lcdm{1,3,10,30,100,300}k.txt" --affinity/lambda=0.9, --affinity/pref=-8134, --threads/n_threads=2
fx-run a-lcdm-1 ./affinity "--data=/local/garryb/lcdm{1000,3000,10000}k.txt" --affinity/lambda=0.98, --affinity/pref=-8134, --threads/n_threads=2

