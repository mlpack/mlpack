#include <unistd.h>
#include <stdio.h>
#include "mpi.h"

main(argc, argv)
int argc;
char** argv;

{
  int my_rank;
  int p;
  int source;
  int dest;
  int silen = 128;
  int gherr;
  char hname[128];
  char message[800];
  MPI_Status status;
  int tag = 50;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  gherr = gethostname( hname, silen);

  if (my_rank != 0) {
    sprintf(message, "Greetings from process %d on %s!", my_rank, hname);
    dest = 0;
    MPI_Send(message, strlen (message)+1, MPI_CHAR, dest,
	     tag, MPI_COMM_WORLD);
  } else {

    printf ("Messages received by process %d on %s.\n\n", my_rank, hname);
    for (source = 1; source < p; source++) {
      MPI_Recv(message, 800, MPI_CHAR, source, tag,
	       MPI_COMM_WORLD, &status);
      printf("%s\n", message);
    }
  }
  MPI_Finalize();
}
