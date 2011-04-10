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

  double lambda_1 = 1.5;//0.4;//0.0001;//0.001;
  double lambda_2 = 0.001;//0.001;//0.05;
  double lambda_w = 0.1;//1.0;//1;//0.1;
  
  
  DiscrSparseCoding dsc;

  /*
  u32 n_dims = 10;
  u32 n_points = 30;
  
  mat X = randn(n_dims, n_points);
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }
  vec y = randn(n_points);
  */
  
  mat X;
  X.load("../contrib/niche/discriminative_sparse_coding/X_hard.dat", raw_ascii);

  //u32 n_dims = X.n_rows;
  u32 n_points = X.n_cols;
  
  vec y;
  y.load("../contrib/niche/discriminative_sparse_coding/y_hard.dat", raw_ascii);
  
  dsc.Init(X, y, n_atoms, lambda_1, lambda_2, lambda_w);
  
  double step_size = 0.1;// not used
  u32 n_iterations = 2000;

  //dsc.SetDictionary(eye(10,10));
  srand ( time(NULL) );
  //dsc.InitDictionary();
  dsc.InitDictionary("InitialDictionary_hard.dat");
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
