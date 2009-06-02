#include "generative_mmk.h"
#include "utils.h"


void TestIsotropicGaussian() {
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
}

void TestHMM() {
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
  PrintDebug("kernel matrix", kernel_matrix, "%3e");
}

void TestKDE() {
  srand48(time(0));

  ArrayList<Matrix> samplings;
  int n_samplings = 10;
  int half_n_samplings = n_samplings / 2;
  n_samplings = 2 * half_n_samplings;
  samplings.Init(n_samplings);
 
  int n_points_per_sampling = 100;
  int n_dims = 1;
 
  for(int k = 0; k < half_n_samplings; k++) {
    Matrix &sampling = samplings[k];
    sampling.Init(n_dims, n_points_per_sampling);
    for(int i = 0; i < n_points_per_sampling; i++) {
      for(int j = 0; j < n_dims; j++) {
	sampling.set(j, i, 2 * drand48());
      }
    }
  }

  for(int k = half_n_samplings; k < n_samplings; k++) {
    Matrix &sampling = samplings[k];
    sampling.Init(n_dims, n_points_per_sampling);
    for(int i = 0; i < n_points_per_sampling; i++) {
      for(int j = 0; j < n_dims; j++) {
	sampling.set(j, i, (2 * drand48()) - ((double)1));
      }
    }
  }


  ScaleSamplingsToCube(&samplings);

  Matrix kernel_matrix;
  KDEGenerativeMMKBatch(1e1, samplings, &kernel_matrix);
  PrintDebug("original kernel matrix", kernel_matrix, "%3e");

  Vector original_eigenvalues;
  la::EigenvaluesInit(kernel_matrix,
		      &original_eigenvalues);
  PrintDebug("original eigenvalues", original_eigenvalues, "%3e");

  NormalizeKernelMatrix(&kernel_matrix);
  PrintDebug("normalized kernel matrix", kernel_matrix, "%3e");
  
  Vector normalized_eigenvalues;
  la::EigenvaluesInit(kernel_matrix,
		      &normalized_eigenvalues);
  PrintDebug("normalized eigenvalues", normalized_eigenvalues, "%3e");
}

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);
  
  //TestIsotropicGaussian();
  //TestHMM();
  TestKDE();

  fx_done(fx_root);
}
