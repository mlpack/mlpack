#include "cosine_tree.h"
#include "quicsvd.h"

#define BOOST_TEST_MODULE QuicSVDTest
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(QuicSVDTest) { 

    FILE * logfile;
    logfile = fopen("LOG", "w");

    Matrix A;
    arma::mat tmpA;
    printf("Load data from input1.txt.\n");
    data::Load("input2.txt", tmpA);
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

    data::Save("U.txt", tmpU);
    data::Save("S.txt", tmpS);
    data::Save("VT.txt", tmpVT);
 }


BOOST_AUTO_TEST_CASE(CosineNodeTest) {
    FILE * logfile;
    logfile = fopen("LOG", "w");

    Matrix A;
    arma::mat tmpA;
    data::Load("input.txt", tmpA);
    arma_compat::armaToMatrix(tmpA, A);
    CosineNode root(A);

    //todo:  figure out what breaks Split()
    //    root.Split();
}










