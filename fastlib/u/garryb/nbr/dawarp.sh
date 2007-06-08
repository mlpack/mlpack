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
mpirun -machinefile warp.txt -np 9 /tmp/allnn_mpi --q=/tmp/qsbig.txt --r=/tmp/qsbig.txt --q/leaf_size=64 --r/leaf_size=64 --n_threads=4 --q/vectors_per_block=$V --q/nodes_per_block=$N --r/vectors_per_block=$V --r/nodes_per_block=$N --monochromatic --n_grains=32 2>&1 | tee dawarp.txt
