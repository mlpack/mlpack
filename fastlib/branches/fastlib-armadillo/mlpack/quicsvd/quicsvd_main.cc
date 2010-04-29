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
#include "quicsvd.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

const fx_entry_doc quicsvd_main_entries[] = {
  {"A_in", FX_REQUIRED, FX_STR, NULL,
   " File consists of matrix A to be decomposed A = U S VT. \n"},
  {"relErr", FX_PARAM, FX_DOUBLE, NULL,
   " Target relative error |A|-|A'|/|A|, default = 0.1.\n"},
  {"U_out", FX_PARAM, FX_STR, NULL,
   " File to hold matrix U.\n"},
  {"s_out", FX_PARAM, FX_STR, NULL,
   " File to hold the singular values vector s.\n"},
  {"VT_out", FX_PARAM, FX_STR, NULL,
   " File to hold matrix VT (V transposed).\n"},
  {"SVT_out", FX_PARAM, FX_STR, NULL,
   " File to hold matrix S * VT (the dimension reduced data).\n"},
  {"lasvd", FX_PARAM, FX_STR, NULL,
   " Use this parameter to compare running time to that of la::SVDInit().\n"},
  {"quicsvd_time", FX_TIMER, FX_CUSTOM, NULL,
   " time to run the QUIC-SVD algorithm.\n"},
  {"lasvd_time", FX_TIMER, FX_CUSTOM, NULL,
   " time to run the SVD algorithm from LAPACK.\n"},
  {"actualErr", FX_RESULT, FX_DOUBLE, NULL,
   " actual relative norm error.\n"},
  {"dimension", FX_RESULT, FX_INT, NULL,
   " the reduced dimension of the data.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc quicsvd_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc quicsvd_main_doc = {
  quicsvd_main_entries, quicsvd_main_submodules,
  "This is a program calculating an approximated Singular "
  "Value Decomposition using QUIC-SVD method.\n"
};

double norm(const Matrix& A) {
  double s = 0;
  for (int i = 0; i < A.n_cols(); i++) {
    Vector col;
    A.MakeColumnVector(i, &col);
    s += math::Sqr(la::LengthEuclidean(col));
  }
  return sqrt(s);
}

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &quicsvd_main_doc);

  Matrix A, U, VT;
  Vector s;

  // parse input file to get matrix A
  const char* A_in = fx_param_str(NULL, "A_in", NULL);

  printf("Loading data ... ");
  fflush(stdout);
  arma::mat tmp;
  data::Load(A_in, tmp);
  arma_compat::armaToMatrix(tmp, A);
  printf("n_rows = %d, n_cols = %d, done.\n", A.n_rows(), A.n_cols());

  // parse target relative error, default = 0.1
  const double targetRelErr = fx_param_double(NULL, "relErr", 0.1);

  printf("QUIC-SVD start ... ");
  fflush(stdout);
  fx_timer_start(NULL, "quicsvd_time");
  // call the QUIC-SVD method
  double actualErr = QuicSVD::SVDInit(A, targetRelErr, &s, &U, &VT);
  fx_timer_stop(NULL, "quicsvd_time");
  printf("stop.\n");
  
  fx_result_double(NULL, "actualErr", actualErr);
  fx_result_int(NULL, "dimension", s.length());

  if (fx_param_exists(NULL, "U_out")) {
    arma_compat::matrixToArma(U, tmp);
    data::Save(fx_param_str(NULL, "U_out", NULL), tmp);
  }
  //else // use OT to write to standard output
  //  ot::Print(U, "U", stdout);

  if (fx_param_exists(NULL, "s_out")) {
    Matrix S;
    S.AliasColVector(s);
    arma_compat::matrixToArma(S, tmp);
    data::Save(fx_param_str(NULL, "s_out", NULL), tmp);
  }
  //else 
  //  ot::Print(s, "s", stdout);

  if (fx_param_exists(NULL, "VT_out")) {
    arma_compat::matrixToArma(VT, tmp);
    data::Save(fx_param_str(NULL, "VT_out", NULL), tmp);
  }
  //else 
  //  ot::Print(VT, "VT", stdout);

  if (fx_param_exists(NULL, "SVT_out")) {
    la::ScaleRows(s, &VT);
    arma_compat::matrixToArma(VT, tmp);
    data::Save(fx_param_str(NULL, "SVT_out", NULL), tmp);
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

  if (fx_param_exists(NULL, "lasvd")) {
    s.Destruct();
    U.Destruct();
    VT.Destruct();
    printf("LAPACK-SVD start ... ");
    fflush(stdout);
    fx_timer_start(NULL, "lasvd_time");
    // call the QUIC-SVD method
    la::SVDInit(A, &s, &U, &VT);
    fx_timer_stop(NULL, "lasvd_time");
    printf("stop.\n");
  }

  fx_done(NULL);
}
