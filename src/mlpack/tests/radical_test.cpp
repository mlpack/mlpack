/**
 * @file radical_main.cpp
 * @author Nishant Mehta
 *
 * Executable for RADICAL
 */
#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/methods/radical/radical.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Radical_Test);


using namespace std;
using namespace arma;



BOOST_AUTO_TEST_CASE(Radical_Test_Radical3D) {
  
  mat matX;
  data::Load("/scratch/niche/mlpack_11_11_11/mlpack/trunk/src/mlpack/tests/data/data_3d_mixed.txt", matX);
  
  mlpack::radical::Radical rad(0.175, 5, 100, matX.n_rows - 1);
  mat matY;
  mat matW;
  
  rad.DoRadical(matX, matY, matW);
  
  
  mat matYT = trans(matY);
  double valEst = 0;
  for(u32 i = 0; i < matYT.n_cols; i++) {
    vec y = vec(matYT.col(i));
    valEst += rad.Vasicek(y);
  }  
  
  mat matS;
  data::Load("/scratch/niche/mlpack_11_11_11/mlpack/trunk/src/mlpack/tests/data/data_3d_ind.txt", matS);
  rad.DoRadical(matS, matY, matW);
  matYT = trans(matY);
  double valBest = 0;
  for(u32 i = 0; i < matYT.n_cols; i++) {
    vec y = vec(matYT.col(i));
    valBest += rad.Vasicek(y);
  }
  
  BOOST_REQUIRE_CLOSE(valBest, valEst, 0.01);
  
}






BOOST_AUTO_TEST_SUITE_END();
