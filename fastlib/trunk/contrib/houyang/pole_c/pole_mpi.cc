// Parallel Online Learning Experiments

#include "pole.h"
#include "mpi.h"

int main(int argc, char *argv[]) {
  srand(time(NULL));

  int numtasks, rank, rc;

  rc = MPI_Init(&argc, &argv);

  if (rc != MPI_SUCCESS) {
    printf("Error init MPI pgoram!\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // do parallel work below
  printf("Number of tasks = %d. My rank = %d.\n", numtasks, rank);

  MPI_Finalize();

}
