#include "fastlib/fastlib.h"


int main(int argc, char* argv[]) {
  srand48(time(0));

  GenMatrix<double> sequence1;
  GenMatrix<double> sequence2;

  sequence1.Init(2, 2);
  sequence2.Init(2, 3);

  for(int i = 0; i < sequence1.n_cols(); i++) {
    for(int j = 0; j < sequence1.n_rows(); j++) {
      sequence1.set(j, i, 0.01 * round(100.0 * drand48()));
    }
  }

  for(int i = 0; i < sequence2.n_cols(); i++) {
    for(int j = 0; j < sequence2.n_rows(); j++) {
      sequence2.set(j, i, 0.01 * round(100.0 * drand48()));
    }
  }

  double lambda = 1;
  
  int order = 2;

  int seq1_length = sequence1.n_cols();
  int seq2_length = sequence2.n_cols();

  int n_dims = sequence1.n_rows();

  int n_features_per_clique = n_dims * order;

  double sum = 0;
  for(int i = 0; i < seq1_length - order + 1; i++) {
    for(int j = 0; j < seq2_length - order + 1; j++) {

      // pick i->(i+order-1) columns of sequence1
      // pick j->(j+order-1) columns of sequence2
      
      //compute squared distance between unrolled vectors (equivalently, subtract submatrices and compute frobenius norm)

      double dist_sq = la::DistanceSqEuclidean(n_features_per_clique, 
					       sequence1.ptr() + (i * n_dims),
					       sequence2.ptr() + (j * n_dims));
      
      // accumulate exp(-lambda * (distance squared)

      sum += exp(-lambda * dist_sq);
    }
  }

  sum /= (double)((seq1_length - order + 1) * (seq2_length - order + 1));


  sequence1.PrintDebug("sequence1");
  sequence2.PrintDebug("sequence2");

  printf("sum = %f\n", sum);
}
