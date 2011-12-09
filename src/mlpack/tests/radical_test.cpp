/**
 * @file radical_main.cpp
 * @author Nishant Mehta
 *
 * Executable for RADICAL
 */
#include <armadillo>
#include <mlpack/methods/radical/radical.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Radical_Test);


using namespace std;
using namespace arma;



BOOST_AUTO_TEST_CASE(Radical_Test_Radical3D) {
  
  mat X;
  X.load("data_3d_mixed");
  
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
  
  mat S;
  S.load("data_3d_ind");
  rad.DoRadical(S, Y, W);
  YT = trans(Y);
  double valBest = 0;
  for(u32 i = 0; i < YT.n_cols; i++) {
    vec Yi = vec(YT.col(i));
    valBest += rad.Vasicek(Yi);
  }
  
  BOOST_REQUIRE_CLOSE(valBest, valEst, 0.01);
  
}






BOOST_AUTO_TEST_SUITE_END();
