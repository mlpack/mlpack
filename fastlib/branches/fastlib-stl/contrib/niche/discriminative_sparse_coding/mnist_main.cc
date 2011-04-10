/** @file main.cc
 *
 *  Driver file for testing LARS
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include "discr_sparse_coding.h"

using namespace arma;
using namespace std;

int main(int argc, char* argv[]) {
  u32 n_atoms = 100;

  double lambda_1 = 0.055;
  double lambda_2 = 0.001;
  double lambda_w = 0.1;

  
  
  
  DiscrSparseCoding dsc;

 
  
  mat X7;
  X7.load("../contrib/niche/discriminative_sparse_coding/mnist/train7.arm");
  u32 n7_points = X7.n_cols;
  n7_points = (int) (n7_points * 0.8);

  mat X9;
  X9.load("../contrib/niche/discriminative_sparse_coding/mnist/train9.arm");
  u32 n9_points = X9.n_cols;
  n9_points = (int) (n9_points * 0.8);
  
  mat X = join_rows(X7(span::all, span(0, n7_points - 1)),
		    X9(span::all, span(0, n9_points - 1)));
  
  u32 n_points = X.n_cols;

  vec y = vec(n_points);
  y.subvec(0, n7_points - 1).fill(-1);
  y.subvec(n7_points, n_points - 1).fill(1);

  printf("%d points\n", n_points);
  
  
  
  // normalize each column of data
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }
  
  X.save("normalized_mnist_7vs9.dat", raw_ascii);
  
  
  
  dsc.Init(X, y, n_atoms, lambda_1, lambda_2, lambda_w);
  
  double step_size = 0.1;// not used
  u32 n_iterations = 20000;
  
  //dsc.SetDictionary(eye(10,10));
  srand ( time(NULL) );
  //dsc.InitDictionary();
  dsc.InitDictionary("MNIST7v9_100atoms_InitialDictionary.dat");
  dsc.InitW();


  
  dsc.SGDOptimize(n_iterations, step_size);

  
  vec w;
  dsc.GetW(w);
  mat D;
  dsc.GetDictionary(D);


  mat V = mat(n_atoms, n_points);
  for(u32 i = 0; i < n_points; i++) {
    Lars lars;
    lars.Init(D, X.col(i), true, lambda_1, lambda_2);
    lars.DoLARS();
    vec v;
    lars.Solution(v);
    V.col(i) = v;
  }

  vec predictions = trans(trans(w) * V);
  vec y_hat = vec(n_points);
  for(u32 i = 0; i < n_points; i++) {
    if(predictions(i) != 0) {
      y_hat(i) =  predictions(i) / fabs(predictions(i));
    }
    else {
      y_hat(i) = 0;
    }
  }

  mat compare = join_rows(y, y_hat);
  compare.print("y y_hat");
  
  double error = 0;
  for(u32 i = 0; i < n_points; i++) {
    if(y(i) != y_hat(i)) {
      error++;
    }
  }
  error /= ((double)n_points);
  printf("error: %f\n", error);
  
  
}
