#include "zhmm.h"
#include "multinomial.h"
#include "diag_gaussian.h"
#include "mixture.h"


void TestMultinomial() {
  HMM<Multinomial> hmm;
  hmm.Init(2, 2, MULTINOMIAL);
  srand48(time(0));
  hmm.RandomlyInitialize();
  
  ArrayList<GenMatrix<int> > sequences;
  sequences.Init(1);
  sequences[0].Init(1, 100);
  for(int i = 0; i < 100; i++) {
    sequences[0].set(0, i, i >= 50);
  }
  //sequences[0].PrintDebug("sequences[0]");
  
  hmm.PrintDebug("hmm");
  hmm.BaumWelch(sequences,
		1e-10 * ((double)1),
		1000);
  hmm.PrintDebug("hmm");
}

void TestGaussian() {
  HMM<DiagGaussian> hmm;
  hmm.Init(2, 2, GAUSSIAN, 0.001);
  srand48(time(0));
  hmm.RandomlyInitialize();
  
  ArrayList<GenMatrix<double> > sequences;
  sequences.Init(1);
  sequences[0].Init(2, 100);
  for(int i = 0; i < 100; i++) {
    double num1 = drand48() / ((double)10);
    double num2 = drand48() / ((double)10);
    if(i >= 50) {
      num1 += ((double)1);
      num2 -= ((double)1);
    }
    sequences[0].set(0, i, num1);
    sequences[0].set(1, i, num2);
  }
  sequences[0].PrintDebug("sequences[0]");
  
  hmm.PrintDebug("hmm");
  hmm.BaumWelch(sequences,
		1e-10 * ((double)1),
		1000);
  hmm.PrintDebug("hmm");
}


void TestMixture() {
  HMM<Mixture<DiagGaussian> > hmm;
  hmm.Init(2, 2, MIXTURE, 0.0001, 2);
  srand48(time(0));
  hmm.RandomlyInitialize();
  
  ArrayList<GenMatrix<double> > sequences;
  sequences.Init(1);
  sequences[0].Init(2, 100);
  for(int i = 0; i < 100; i++) {
    double num1 = drand48() / ((double)10);
    double num2 = drand48() / ((double)10);
    if(i >= 50) {
      num1 += ((double)1);
      num2 -= ((double)1);
    }
    sequences[0].set(0, i, num1);
    sequences[0].set(1, i, num2);
  }
  sequences[0].PrintDebug("sequences[0]");
  
  hmm.PrintDebug("hmm");
  hmm.BaumWelch(sequences,
		1e-10 * ((double)1),
		1000);
  hmm.PrintDebug("hmm");
}


void TestGaussianPdf() {
  DiagGaussian g;
  g.Init(2, 0.001);
  Vector &mu = *(g.mu_);
  mu[0] = -1;
  mu[1] = 0.5;

  Vector &sigma = *(g.sigma_);
  sigma[0] = 0.5;
  sigma[1] = 0.4;
  
  g.ComputeNormConstant();
  g.PrintDebug("g");
  
  Vector x;
  x.Init(2);
  x[0] = 2;
  x[1] = 1;
  printf("%e\n", g.Pdf(x));


}


int main(int argc, char* argv[]) {
  //TestMultinomial();
  TestGaussian();
  //TestGaussianPdf();
  //TestMixture();
}
