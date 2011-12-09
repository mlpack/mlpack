/**
 * @file radical_main.cpp
 * @author Nishant Mehta
 *
 * Executable for RADICAL
 */
#include <armadillo>
#include "radical.hpp"

using namespace std;
using namespace arma;



void test() {
  
  mat X;
  X.load("/net/hu15/niche/matlab/toolboxes/RADICAL/examples/data_3d_mixed");
  
  mlpack::radical::Radical rad(0.175, 5, 100, X.n_rows - 1);
  mat Y;
  mat W;
  
  rad.DoRadical(X, Y, W);
  
  
  mat YT = trans(Y);
  double valEst = 0;
  for(u32 i = 0; i < YT.n_cols; i++) {
    vec Yi = vec(YT.col(i));
    valEst += rad.Vasicek(Yi);
  }
  printf("objective(estimate) = %f\n", valEst);


  
  
  mat S;
  S.load("/net/hu15/niche/matlab/toolboxes/RADICAL/examples/data_3d_ind");
  rad.DoRadical(S, Y, W);
  YT = trans(Y);
  double valBest = 0;
  for(u32 i = 0; i < YT.n_cols; i++) {
    vec Yi = vec(YT.col(i));
    valBest += rad.Vasicek(Yi);
  }
  printf("objective(sources) = %f\n", valBest);
  
  
  
  
    
}





int main(int argc, char* argv[]) {
  test();
  
  return 1;
  
  size_t nPoints = 1000;
  size_t nDims = 2;

  mat S = randu(nDims, nPoints);
  //S.print("S");
  mat Mixing = randn(nDims, nDims);
  Mixing.print("Mixing");
  mat X = Mixing * S;
  
  /*
  mat U, V;
  vec s;
  svd(U, s, V, cov(X));
  mat Whitening = U * diagmat(pow(s, -0.5)) * trans(V);
  X = X * Whitening;
  */

  
  mlpack::radical::Radical rad(0.01, 10, 200, 1);

  mat Y;
  mat W;
  rad.DoRadical(X, Y, W);
  
  W.print("W");
  
  X = trans(X);
  mat XWhitened;
  mat Whitening;
  mlpack::radical::WhitenFeatureMajorMatrix(X, XWhitened, Whitening);
  
  double val_init = 0;
  for(size_t i = 0; i < nDims; i++) {
    //val_init += rad.Vasicek(XWhitened.col(i));
  }
  printf("initial objective value: %f\n", val_init);
  
  Y = trans(Y);
  double val_final = 0;
  for(size_t i = 0; i < nDims; i++) {
    //val_final += rad.Vasicek(Y.col(i));
  }
  printf("final objective value: %f\n", val_final);
  printf("improvement: %f\n", val_init - val_final);


}
