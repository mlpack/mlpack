#include "cosine_tree.h"
#include "quicsvd.h"

#define BOOST_TEST_MODULE QuicSVDTest
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(QuicSVDTest) { 

  Matrix A;
  arma::mat tmpA = "0.0 1.0 2.0 3.0; 0.0 1.0 2.0 3.0";
  arma_compat::armaToMatrix(tmpA, A);
  QuicSVD svd(A, 0.1);
  Vector s;
  Matrix U, VT,S;

  svd.ComputeSVD(&s, &U, &VT);
  arma::mat tmpU, tmpVT, tmpS;
  arma_compat::matrixToArma(U, tmpU);
  arma_compat::matrixToArma(VT, tmpVT);
  S.InitDiagonal(s);
  arma_compat::matrixToArma(S, tmpS);

  // There is nobody to check this, so don't save it...
//    data::Save("U.txt", tmpU);
//    data::Save("S.txt", tmpS);
//    data::Save("VT.txt", tmpVT);
}

BOOST_AUTO_TEST_CASE(CosineNodeTest) {
  Matrix A;
  arma::mat tmpA = "0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0";
  arma_compat::armaToMatrix(tmpA, A);
  CosineNode root(A);

  //todo:  figure out what breaks Split()
  //    root.Split();
}
