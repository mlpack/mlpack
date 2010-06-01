#include "latent_mmk.h"
#include "utils.h"


void PrintDebugGenMatrixInt(const char* name, GenMatrix<int> x) {
  int n_rows = x.n_rows();
  int n_cols = x.n_cols();
  printf("----- GENMATRIX<INT> %s ------\n", name);
  for(int i = 0; i < n_rows; i++) {
    for(int j = 0; j < n_cols; j++) {
      printf("%d ", x.get(i, j));
    }
  }
  printf("\n");
}

void PrintIntData(const char* name,
	       const ArrayList<GenMatrix<int> > &data) {
  printf("----- ARRAYLIST<GENMATRIX<INT> > ------\n");
  for(int i = 0; i < data.size(); i++) {
    char matrix_name[80];
    sprintf(matrix_name, "%s[%d]", name, i);
    PrintDebugGenMatrixInt(matrix_name, data[i]);
  }
}

    

int main(int argc, char* argv[]) {

  const char* filename = "../hshmm/exons_small.dat";
  
  ArrayList<GenMatrix<int> > sequences;
  LoadVaryingLengthData(filename, &sequences);
  
  //PrintIntData("sequences", sequences);

  HMM<Multinomial> hmm;
  hmm.Init(20, 4, MULTINOMIAL);
  srand48(time(0));
  //hmm.RandomlyInitialize();
  hmm.InitParameters(sequences);
  hmm.PrintDebug("hmm after calling InitParameters(sequences)");
  hmm.ViterbiUpdate(sequences);
  hmm.PrintDebug("hmm after calling ViterbiUpdate(sequences)");
  hmm.BaumWelch(sequences,
		1e-10 * ((double)1),
		5);
  hmm.PrintDebug("hmm after calling BaumWelch");

  double lmmk_1_2 = LatentMMK(1, hmm, sequences[0], sequences[1]);
  printf("lmmk(sequences[0], sequences[1] = %f\n", lmmk_1_2);
  
  Matrix kernel_matrix;
  LatentMMKBatch(1, hmm, sequences, &kernel_matrix);
  kernel_matrix.PrintDebug("kernel matrix");

}
