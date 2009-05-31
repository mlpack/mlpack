#include "fastlib/fastlib.h"
#include "empirical_mmk.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  srand48(time(0));
  srand(time(0));

  GenMatrix<int> sequence1;
  sequence1.Init(1, 10);
  
  sequence1.set(0, 0, 0);
  sequence1.set(0, 1, 1);
  sequence1.set(0, 2, 2);
  sequence1.set(0, 3, 3);
  sequence1.set(0, 4, 0);
  sequence1.set(0, 5, 1);
  sequence1.set(0, 6, 1);
  sequence1.set(0, 7, 3);
  sequence1.set(0, 8, 0);
  sequence1.set(0, 9, 1);
 
 
  GenMatrix<int> sequence2;
  sequence2.Init(1, 10);
  sequence2.set(0, 0, 0);
  sequence2.set(0, 1, 1);
  sequence2.set(0, 2, 2);
  sequence2.set(0, 3, 3);
  sequence2.set(0, 4, 0);
  sequence2.set(0, 5, 1);
  sequence2.set(0, 6, 1);
  sequence2.set(0, 7, 3);
  sequence2.set(0, 8, 0);
  sequence2.set(0, 9, 1);

  GenVector<int> counts1;
  GetCounts(1, 4, sequence1, &counts1);
  PrintDebug("sequence 1 counts", counts1, "%d");

  GenVector<int> counts2;
  GetCounts(1, 4, sequence2, &counts2);
  PrintDebug("sequence 2 counts", counts2, "%d");

  double lambda = 1e9;

  printf("memmk = %f\n",
	 MarkovEmpiricalMMK(lambda, 1, 4, sequence1, sequence2));


  fx_done(fx_root);



}
