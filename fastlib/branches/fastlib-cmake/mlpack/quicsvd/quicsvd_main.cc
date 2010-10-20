/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
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
#include <boost/program_options.hpp>

using namespace std;
namespace boost_po = boost::program_options;
boost_po::variables_map vm;

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
  //const char* A_in = fx_param_str(NULL, "A_in", NULL);
  boost_po::options_description desc("Allowed options");
  desc.add_options()
      ("A_in", boost_po::value<char *>(), "  File consists of matrix A to be decomposed A = U S VT. \n")
      ("relErr", boost_po::value<double>(), "  Target relative error |A| - |A'|/|A|, default = 0.1\n")
      ("U_out", boost_po::value<char *>(), " File to hold matrix U.\n")
      ("s_out", boost_po::value<char *>(), " File to hold the singular values vector s. \n")
      ("VT_out", boost_po::value<char *>(), " File to hold matrix VT (V transposed). \n")
      ("SVT_out", boost_po::value<char *>(), "File to hold matrix S * VT (the dimension reduced data). \n")
      ("lasvd", boost_po::value<char *>(), " Use this parameter to compare running time to that of la::SVDInit()");

  boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
  boost_po::notify(vm);

  const char* A_in = vm["A_in"].as<char *>();

  printf("Loading data ... ");
  fflush(stdout);
  data::Load(A_in, &A);
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

//  if (fx_param_exists(NULL, "U_out"))
  if ( 0 != vm.count("U_out"))
    data::Save(fx_param_str(NULL, "U_out", NULL), U);
  //else // use OT to write to standard output
  //  ot::Print(U, "U", stdout);

//  if (fx_param_exists(NULL, "s_out")) {
  if ( 0 != vm.count("s_out"))
  {
    Matrix S;
    S.AliasColVector(s);
    //data::Save(fx_param_str(NULL, "s_out", NULL), S);
    data::Save(vm["s_out"].as<char *>(), S);
  }
  //else 
  //  ot::Print(s, "s", stdout);

//  if (fx_param_exists(NULL, "VT_out"))
  if ( 0 != vm.count("VT_out"))
  {
   // data::Save(fx_param_str(NULL, "VT_out", NULL), VT);
    data::Save(vm["VT_out"].as<char *>(), VT);
  }
  //else 
  //  ot::Print(VT, "VT", stdout);

//  if (fx_param_exists(NULL, "SVT_out")) {
  if ( 0 != vm.count("SVT_out")) {
    la::ScaleRows(s, &VT);
    //data::Save(fx_param_str(NULL, "SVT_out", NULL), VT);
    data::Save(vm["SVT_out"].as<char *>(), VT);
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

  //if (fx_param_exists(NULL, "lasvd")) {
  if ( 0 != vm.count("lasvd")) {
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
