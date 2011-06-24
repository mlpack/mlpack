/**
 * @file quicsvd_main.cc
 *
 * This file implements command line interface for the QUIC-SVD
 * method. It approximate the original matrix by another matrix
 * with smaller dimension to a certain accuracy degree specified by the 
 * user and then make SVD decomposition in the projected supspace.
 *
 * Run with --help for more usage.
 * 
 * @see quicsvd.h
 */

#include <fastlib/fastlib.h>
#include <fastlib/fx/io.h>
#include "quicsvd.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

PARAM_STRING_REQ("A_in", "File consists of matrix A to be decomposed A = U S VT.", "kernel");
PARAM_STRING("U_out", "File to hold matrix U.", "kernel", "");
PARAM_STRING("s_out", "File to hold the singular values vector s.", "kernel", "");
PARAM_STRING("VT_out", "File to hold matrix VT (V transposed).", "kernel", "");
PARAM_STRING("SVT_out", "File to hold matrix S * VT (the dimension reduced data).", "kernel", "");
PARAM_STRING("lasvd", "Use this parameter to compare running time to that of la::SVDInit().", "kernel", "");

PARAM_INT("dimension", "the reduced dimension of the data.", "kernel", 0);

PARAM(double, "actualErr", "actual relative norm error.", "kernel", 0.0, false);
PARAM(double, "relErr", "actual relative norm error.", "kernel", 0.1, false);

PROGRAM_INFO("QuicSVD", "This is a program calculating an approximated\
 Singular Value Decomposition using QUIC-SVD method.", "kernel");

using namespace mlpack;

double norm(const Matrix& A) {
  double s = 0;
  for (index_t i = 0; i < A.n_cols(); i++) {
    Vector col;
    A.MakeColumnVector(i, &col);
    s += math::Sqr(la::LengthEuclidean(col));
  }
  return sqrt(s);
}

int main(int argc, char* argv[]) {
  IO::ParseCommandLine(argc, argv);

  Matrix A, U, VT;
  Vector s;

  // parse input file to get matrix A
  const char* A_in = IO::GetParam<std::string>("kernel/A_in").c_str();

  IO::Info << "Loading data ... " << std::endl;
  fflush(stdout);
  arma::mat tmp;
  data::Load(A_in, tmp);
  arma_compat::armaToMatrix(tmp, A);
  IO::Info << "n_rows = " << A.n_rows() << ", n_cols = " << 
    A.n_cols() << ", done." << std::endl;

  // parse target relative error, default = 0.1
  const double targetRelErr = IO::GetParam<double>("kernel/relErr");

  IO::Info << "QUIC-SVD start ... " << std::endl;
  fflush(stdout);
  IO::StartTimer("kernel/quicsvd_time");
  // call the QUIC-SVD method
  double actualErr = QuicSVD::SVDInit(A, targetRelErr, &s, &U, &VT);
  IO::StopTimer("kernel/quicsvd_time");
  IO::Info << "stop." << std::endl;
  
  IO::GetParam<double>("kernel/actualErr") = actualErr;
  IO::GetParam<int>("kernel/dimension") = s.length();

  if (IO::HasParam("kernel/U_out")) {
    arma_compat::matrixToArma(U, tmp);
    data::Save(IO::GetParam<std::string>("kernel/U_out").c_str(), tmp);
  }
  //else // use OT to write to standard output
  //  ot::Print(U, "U", stdout);

  if (IO::HasParam("kernel/s_out")) {
    Matrix S;
    S.AliasColVector(s);
    arma_compat::matrixToArma(S, tmp);
    data::Save(IO::GetParam<std::string>("kernel/s_out").c_str(), tmp);
  }
  //else 
  //  ot::Print(s, "s", stdout);

  if (IO::HasParam("kernel/VT_out")) {
    arma_compat::matrixToArma(VT, tmp);
    data::Save(IO::GetParam<std::string>("kernel/VT_out").c_str(), tmp);
  }
  //else 
  //  ot::Print(VT, "VT", stdout);

  if (IO::HasParam("kernel/SVT_out")) {
    la::ScaleRows(s, &VT);
    arma_compat::matrixToArma(VT, tmp);
    data::Save(IO::GetParam<std::string>("kernel/SVT_out").c_str(), tmp);
  }

  /*
  Matrix B, V;
  la::TransposeInit(VT, &V);
  B.Init(A.n_rows(), A.n_cols());
  B.SetZero();

  for (index_t i = 0; i < s.length(); i++) {
    Vector ucol, vcol;
    U.MakeColumnVector(i, &ucol);
    V.MakeColumnVector(i, &vcol);
    Matrix ucol_i, vcol_i;
    ucol_i.AliasColVector(ucol);
    vcol_i.AliasColVector(vcol);
    la::MulExpert(s[i], false, ucol_i, true, vcol_i, 1, &B);
  }

  la::SubFrom(A, &B);

  printf("relative error: %f\n", norm(B)/norm(A));
  */

  if (IO::HasParam("kernel/lasvd")) {
    s.Destruct();
    U.Destruct();
    VT.Destruct();
    IO::Info << "LAPACK-SVD start ... " << std::endl;
    fflush(stdout);
    IO::StartTimer("kernel/lasvd_time");
    // call the QUIC-SVD method
    la::SVDInit(A, &s, &U, &VT);
    IO::StopTimer("kernel/lasvd_time");
    IO::Info << "stop." << std::endl;
  }
}
