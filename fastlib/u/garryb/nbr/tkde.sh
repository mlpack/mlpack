V="$1"
N="$2"
mpirun -np 3 -machinefile intel.txt ./tkde_mpi --q=/local/garryb/qsbig.txt --r=/local/garryb/qsbig.txt --tkde/h=0.15 --tkde/threshold=22000000 --n_threads=2 --q/vectors_per_block=$V --q/nodes_per_block=$N --r/vectors_per_block=$V --r/nodes_per_block=$N | grep all_threads
