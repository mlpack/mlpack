#include "latent_mmk.h"

int main(int argc, char* argv[]) {
  HMM<Multinomial> hmm;
  hmm.Init(2, 2, MULTINOMIAL);
  srand48(time(0));
  hmm.RandomlyInitialize();
  
  ArrayList<GenMatrix<int> > sequences;
  sequences.Init(2);
  sequences[0].Init(1, 100);
  for(int i = 0; i < 100; i++) {
    sequences[0].set(0, i, i >= 50);
  }
  sequences[1].Init(1, 100);
  for(int i = 0; i < 100; i++) {
    sequences[1].set(0, i, i < 50);
  }
  
  printf("sequences[0]\n");
  for(int i = 0; i < 100; i++) {
    printf("%d ", sequences[0].get(0, i));
  }
  printf("\n");

  printf("sequences[1]\n");
  for(int i = 0; i < 100; i++) {
    printf("%d ", sequences[1].get(0, i));
  }
  printf("\n");
  
  
  hmm.InitParameters(sequences);
  
  hmm.PrintDebug("hmm after calling InitParameters(sequences)");
  
  hmm.ViterbiUpdate(sequences);
  
  hmm.PrintDebug("hmm after calling ViterbiUpdate(sequences)");
  
  hmm.BaumWelch(sequences,
		1e-10 * ((double)1),
		1000);
  
  hmm.PrintDebug("hmm after calling BaumWelch");

  double lmmk_1_2 = LatentMMK(1, hmm, sequences[0], sequences[1]);
  printf("lmmk(seq1, seq2 = %f\n", lmmk_1_2);

  Matrix kernel_matrix;
  LatentMMKBatch(1, hmm, sequences, &kernel_matrix);
  kernel_matrix.PrintDebug("kernel matrix");
}
