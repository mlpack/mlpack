#include "fastlib/fastlib.h"
#include "empirical_mmk.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  srand48(time(0));
  srand(time(0));

  ArrayList<GenMatrix<int> > sequences;
  sequences.Init(2);
  sequences[0].Init(1, 10);
  
  sequences[0].set(0, 0, 0);
  sequences[0].set(0, 1, 1);
  sequences[0].set(0, 2, 2);
  sequences[0].set(0, 3, 3);
  sequences[0].set(0, 4, 0);
  sequences[0].set(0, 5, 1);
  sequences[0].set(0, 6, 1);
  sequences[0].set(0, 7, 3);
  sequences[0].set(0, 8, 0);
  sequences[0].set(0, 9, 1);
 
  sequences[1].Init(1, 8);
  sequences[1].set(0, 0, 0);
  sequences[1].set(0, 1, 1);
  sequences[1].set(0, 2, 1);
  sequences[1].set(0, 3, 2);
  sequences[1].set(0, 4, 2);
  sequences[1].set(0, 5, 3);
  sequences[1].set(0, 6, 0);
  sequences[1].set(0, 7, 1);

  GenVector<int> counts1;
  GetCounts(1, 4, sequences[0], &counts1);
  PrintDebug("sequence 1 counts", counts1, "%d");

  GenVector<int> counts2;
  GetCounts(1, 4, sequences[1], &counts2);
  PrintDebug("sequence 2 counts", counts2, "%d");

  double lambda = 1e9;

  printf("memmk = %f\n",
	 MarkovEmpiricalMMK(lambda, 1, 4, sequences[0], sequences[1]));

  Matrix kernel_matrix;
  MarkovEmpiricalMMKBatch(lambda, 1, 4,
			  sequences,
			  &kernel_matrix);
  
  kernel_matrix.PrintDebug("kernel matrix");

  fx_done(fx_root);



}
