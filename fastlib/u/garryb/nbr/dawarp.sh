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

if [ -z "$V" ]; then
  V=256
fi

if [ -z "$N" ]; then
  N=64
fi

mpirun -machinefile warp.txt -np 2 \
    ./allnn_mpi \
    --data=/tmp/a1000k.txt \
    --n_threads=1 \
    --n_grains=32 \
    --data/leaf_size=64 \
    --points_per_block=$V \
    --nodes_per_block=$N \
    2>&1 | tee dawarp.txt
