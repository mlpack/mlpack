/**
 * @file radical_main.cpp
 * @author Nishant Mehta
 *
 * Executable for RADICAL
 */
#include <armadillo>
#include "radical.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  Radical rad;

  u32 nPoints = 1000;
  u32 nDims = 2;

  mat S = randu(nPoints, nDims);
  //S.print("S");
  mat Mixing = randn(nDims, nDims);
  Mixing.print("Mixing");
  mat X = S * Mixing;

  /*
  mat U, V;
  vec s;
  svd(U, s, V, cov(X));
  mat Whitening = U * diagmat(pow(s, -0.5)) * trans(V);
  X = X * Whitening;
  */

  rad.Init(0.01, 10, 200, 1, X);
  mat Y;
  mat W;
  rad.DoRadical(Y, W);
  
  W.print("W");

  double val_init = 0;
  for(u32 i = 0; i < nDims; i++) {
    val_init += rad.Vasicek(rad.GetX().col(i));
  }
  printf("initial objective value: %f\n", val_init);

  
  double val_final = 0;
  for(u32 i = 0; i < nDims; i++) {
    val_final += rad.Vasicek(Y.col(i));
  }
  printf("final objective value: %f\n", val_final);
  printf("improvement: %f\n", val_init - val_final);


}
