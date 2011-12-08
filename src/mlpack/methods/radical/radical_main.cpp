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

int main(int argc, char* argv[]) {
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
  printf("doing radical\n");
  rad.DoRadical(X, Y, W);
  printf("radical done\n");
  
  W.print("W");
  
  X = trans(X);
  mat XWhitened;
  mat Whitening;
  mlpack::radical::WhitenFeatureMajorMatrix(X, XWhitened, Whitening);
  
  double val_init = 0;
  for(size_t i = 0; i < nDims; i++) {
    val_init += rad.Vasicek(XWhitened.col(i));
  }
  printf("initial objective value: %f\n", val_init);
  
  Y = trans(Y);
  double val_final = 0;
  for(size_t i = 0; i < nDims; i++) {
    val_final += rad.Vasicek(Y.col(i));
  }
  printf("final objective value: %f\n", val_final);
  printf("improvement: %f\n", val_init - val_final);


}
