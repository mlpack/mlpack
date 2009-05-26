#include "generative_mmk.h"

int main(int argc, char* argv[]) {

  IsotropicGaussian g1;
  g1.Init(2, 0.001);
  Vector &mu1 = *(g1.mu_);
  mu1[0] = -1;
  mu1[1] = 0.5;
  g1.sigma_ = 0.5;

  IsotropicGaussian g2;
  g2.Init(2, 0.001);
  Vector &mu2 = *(g2.mu_);
  mu2[0] = -1;
  mu2[1] = 0.5;
  g2.sigma_ = 0.5;

  printf("gmmk(isotropic_gaussian1, isotropic_gaussian2) = %f\n",
	 GenerativeMMK(1, g1, g2));
  


  


  ArrayList<HMM<Multinomial> > hmms;
  hmms.Init(2);

  ArrayList<ArrayList<GenMatrix<int> > > sequences;
  sequences.Init(2);

  for(int i = 0; i < 2; i++) {
    sequences[i].Init(1);
    sequences[i][0].Init(1, 100);
  }

  for(int i = 0; i < 100; i++) {
    sequences[0][0].set(0, i, i >= 50);
  }
  for(int i = 0; i < 100; i++) {
    sequences[1][0].set(0, i, i < 50);
  }
  
  for(int i = 0; i < 2; i++) {
    printf("sequences[%d]\n", i);
    for(int t = 0; t < 100; t++) {
      printf("%d ", sequences[i][0].get(0, t));
    }
    printf("\n");
  }

  srand48(time(0));  
  for(int i = 0; i < 2; i++) {
    hmms[i].Init(2, 2, MULTINOMIAL);

    hmms[i].InitParameters(sequences[i]);
  
    hmms[i].PrintDebug("hmm after calling InitParameters(sequences)");
    
    hmms[i].ViterbiUpdate(sequences[i]);
    
    hmms[i].PrintDebug("hmm after calling ViterbiUpdate(sequences)");
  
    hmms[i].BaumWelch(sequences[i],
		      1e-10 * ((double)1),
		      1000);
    
    hmms[i].PrintDebug("hmm after calling BaumWelch");
  }

  double gmmk_1_2 = GenerativeMMK(1, 100, hmms[0], hmms[1]);
  printf("gmmk(hmm1, hmm2) = %f\n", gmmk_1_2);
  
  Matrix kernel_matrix;
  GenerativeMMKBatch(1, 100, hmms, &kernel_matrix);
  kernel_matrix.PrintDebug("kernel matrix");
}
