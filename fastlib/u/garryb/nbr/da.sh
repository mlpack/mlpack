echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"
echo "FIND SOME NEW DATASETS LOSER"




V="$1"
N="$2"
mpirun -np 9 -machinefile intel.txt ./allnn_mpi --q=/local/garryb/qsbig.txt --r=/local/garryb/qsbig.txt --q/leaf_size=32 --r/leaf_size=32 --n_threads=2 --q/vectors_per_block=$V --q/nodes_per_block=$N --r/vectors_per_block=$V --r/nodes_per_block=$N | grep all_threads
