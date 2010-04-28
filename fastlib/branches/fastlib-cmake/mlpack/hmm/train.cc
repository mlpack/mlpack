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
 * @file train.cc
 *
 * This file contains the program to estimate Hidden Markov Model parameter
 * using training sequences.
 *
 * It use two algorithm: Baum-Welch (EM) and Viterbi
 *
 * Usage:
 *   train --type=TYPE --profile=PROFILE --seqfile=FILE [OPTIONS]
 * See the usage() function for complete option list
 */

#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"
#include <iostream>
#include <boost/program_options.hpp>

namespace boost_po = boost::program_options;
boost_po::variables_map vm;

using namespace std;
using namespace hmm_support;

success_t train_baumwelch();
success_t train_viterbi();
void usage();

const fx_entry_doc hmm_train_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"algorithm", FX_PARAM, FX_STR, NULL,
   "  Training algoritm: baumwelch | viterbi.\n"},
  {"seqfile", FX_REQUIRED, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  {"guess", FX_PARAM, FX_STR, NULL,
   "  File containing guessing HMM model profile.\n"},
  {"numstate", FX_PARAM, FX_INT, NULL,
   "  If no guessing profile specified, at least provide the number of states.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  Output file containing trained HMM profile.\n"},
  {"maxiter", FX_PARAM, FX_INT, NULL,
   "  Maximum number of iterations, default = 500.\n"},
  {"tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "  Error tolerance on log-likelihood as a stopping criteria.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_train_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_train_main_doc = {
  hmm_train_main_entries, hmm_train_main_submodules,
  "This is a program training HMM models from data sequences. \n"
};

void usage() {
  printf("\nUsage:\n"
	 "  train --type=={discrete|gaussian|mixture} OPTION\n"
	 "[OPTIONS]\n"
	 "  --algorithm={baumwelch|viterbi} : algorithm used for training, default Baum-Welch\n"
	 "  --seqfile=file   : file contains input sequences\n"
	 "  --guess=file     : file contains guess HMM profile\n"
	 "  --numstate=NUM   : if no guess profile is specified, at least specify the number of state\n"
	 "  --profile=file   : output file for estimated HMM profile\n"
	 "  --maxiter=NUM    : maximum number of iteration, default=500\n"
	 "  --tolerance=NUM  : error tolerance on log-likelihood, default=1e-3\n"
	 );
}

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_train_main_doc);
  success_t s = SUCCESS_PASS;

  boost_po::options_description desc("Allowed options");
  desc.add_options()
      ("type", boost_po::value<std::string>(), "  HMM type : discrete | gaussian | mixture.\n")
      ("algorithm", boost_po::value<std::string>()->default_value("baumwelch"), "  Training algoritm: baumwelch | viterbi.\n")
      ("seqfile", boost_po::value<std::string>(), "  Output file for the data sequences.\n")
      ("guess", boost_po::value<std::string>(), "  File containing guessing HMM model profile.\n")
      ("num_state", boost_po::value<int>(), "  If no guessing profile specified, at least provide the number of states.\n")
      ("profile", boost_po::value<std::string>()->default_value("pro.mix.out"), "  Output file containing trained HMM profile.\n")
      ("maxiter", boost_po::value<int>()->default_value(500), "  Maximum number of iterations, default = 500.\n")
      ("tolerance", boost_po::value<double>()->default_value(0.001), "  Error tolerance on log-likelihood as a stopping criteria.\n");

  boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
  boost_po::notify(vm);

//  if (fx_param_exists(NULL,"type")) {
   if ( 0 != vm.count("type")) {
    //const char* algorithm = fx_param_str(NULL, "algorithm", "baumwelch");
    std::string algorithm = vm["algorithm"].as<std::string>();
    if (strcmp(algorithm.c_str(),"baumwelch")==0)
      s = train_baumwelch();
    else if (strcmp(algorithm.c_str(),"viterbi")==0)
      s = train_viterbi();
    else {
      printf("Unrecognized algorithm: must be baumwelch or viterbi !!!\n");
      s = SUCCESS_FAIL;
    }
  }
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture  !!!\n");
    s = SUCCESS_FAIL;
  }
  if (!PASSED(s)) usage();
  fx_done(NULL);
}

success_t train_baumwelch_discrete();
success_t train_baumwelch_gaussian();
success_t train_baumwelch_mixture();

success_t train_baumwelch() {
  //const char* type = fx_param_str_req(NULL, "type");
  if ( 0 == vm.count("type")) {
    exit(1);
  }
  std::string type = vm["type"].as<std::string>();
  if (strcmp(type.c_str(), "discrete")==0)
    return train_baumwelch_discrete();
  else if (strcmp(type.c_str(), "gaussian")==0)
    return train_baumwelch_gaussian();
  else if (strcmp(type.c_str(), "mixture")==0)
    return train_baumwelch_mixture();
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
    return SUCCESS_FAIL;
  }
}

success_t train_viterbi_discrete();
success_t train_viterbi_gaussian();
success_t train_viterbi_mixture();

success_t train_viterbi() {
  //const char* type = fx_param_str_req(NULL, "type");
  if ( 0 == vm.count("type")) {
    exit(1);
  }
  std::string type = vm["type"].as<std::string>();

  if (strcmp(type.c_str(), "discrete")==0)
    return train_viterbi_discrete();
  else if (strcmp(type.c_str(), "gaussian")==0)
    return train_viterbi_gaussian();
  else if (strcmp(type.c_str(), "mixture")==0)
    return train_viterbi_mixture();
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
    return SUCCESS_FAIL;
  }
}

success_t train_baumwelch_mixture() {
//  if (!fx_param_exists(NULL, "seqfile")) {
  if ( 0 == vm.count("seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }

  MixtureofGaussianHMM hmm;
  ArrayList<Matrix> seqs;

  //const char* seqin = fx_param_str_req(NULL, "seqfile");
  //const char* proout = fx_param_str(NULL, "profile", "pro.mix.out");
  std::string seqin = vm["seqfile"].as<std::string>();
  std::string proout = vm["profile"].as<std::string>();

  load_matrix_list(seqin.c_str(), &seqs);

//  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
  if ( 0 != vm.count("guess")) {
//    const char* guess = fx_param_str_req(NULL, "guess");
    if ( 0 == vm.count("guess")) {
      exit(1);
    }
    std::string guess = vm["guess"].as<std::string>();

    printf("Load parameters from file %s\n", guess.c_str());
    hmm.InitFromFile(guess.c_str());
  }
  else {
    hmm.Init();
    printf("Automatic initialization not supported !!!");
    return SUCCESS_FAIL;
  }

  //int maxiter = fx_param_int(NULL, "maxiter", 500);
  //double tol = fx_param_double(NULL, "tolerance", 1e-3);

  int maxiter = vm["maxiter"].as<int>();
  double tol = vm["tolerance"].as<double>();

  hmm.TrainBaumWelch(seqs, maxiter, tol);

  hmm.SaveProfile(proout.c_str());

  return SUCCESS_PASS;
}

success_t train_baumwelch_gaussian() {
//  if (!fx_param_exists(NULL, "seqfile")) {
   if ( 0 == vm.count("seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }
  GaussianHMM hmm;
  ArrayList<Matrix> seqs;

  //const char* seqin = fx_param_str_req(NULL, "seqfile");
  //const char* proout = fx_param_str(NULL, "profile", "pro.gauss.out");
  std::string seqin = vm["seqfile"].as<std::string>();
  std::string proout = vm["profile"].as<std::string>();

  load_matrix_list(seqin.c_str(), &seqs);

//  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    if ( 0 != vm.count("guess")) {
   //    const char* guess = fx_param_str_req(NULL, "guess");
    if ( 0 == vm.count("guess")) {
      exit(1);
    }
    
    std::string guess = vm["guess"].as<std::string>();
    printf("Load parameters from file %s\n", guess.c_str());
    hmm.InitFromFile(guess.c_str());
  }
  else { // otherwise initialized using information from the data
    //int numstate = fx_param_int_req(NULL, "numstate");
    if ( 0 == vm.count("numstate")) {
       cerr << "Required parameter numstate not specified" << endl;
       exit(1);
    }
    int numstate = vm["numstate"].as<int>();

    printf("Generate HMM parameters: NUMSTATE = %d\n", numstate);
    hmm.InitFromData(seqs, numstate);
    printf("Done.\n");
  }

  //int maxiter = fx_param_int(NULL, "maxiter", 500);
  //double tol = fx_param_double(NULL, "tolerance", 1e-3);

  int maxiter = vm["maxiter"].as<int>();
  double tol = vm["tolerance"].as<double>();

  printf("Training ...\n");
  hmm.TrainBaumWelch(seqs, maxiter, tol);
  printf("Done.\n");

  hmm.SaveProfile(proout.c_str());

  return SUCCESS_PASS;
}

success_t train_baumwelch_discrete() {
//  if (!fx_param_exists(NULL, "seqfile")) {
  if ( 0 == vm.count("seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }

//  const char* seqin = fx_param_str_req(NULL, "seqfile");
//  const char* proout = fx_param_str(NULL, "profile", "pro.dis.out");
  std::string seqin = vm["seqfile"].as<std::string>();
  std::string proout = vm["profile"].as<std::string>();

  ArrayList<Vector> seqs;
  load_vector_list(seqin.c_str(), &seqs);

  DiscreteHMM hmm;

//  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
   if ( 0 != vm.count("guess")) {
     if ( 0 == vm.count("guess")) {
       exit(1);
     }

     std::string guess = vm["guess"].as<std::string>();

    printf("Load HMM parameters from file %s\n", guess.c_str());
    hmm.InitFromFile(guess.c_str());
  }
  else { // otherwise randomly initialized using information from the data
    //int numstate = fx_param_int_req(NULL, "numstate");
    if ( 0 == vm.count("numstate")) {
       cerr << "Required parameter numstate not specified" << endl;
       exit(1);
    }
    int numstate = vm["numstate"].as<int>();

    printf("Randomly generate parameters: NUMSTATE = %d\n", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  //int maxiter = fx_param_int(NULL, "maxiter", 500);
  //double tol = fx_param_double(NULL, "tolerance", 1e-3);

  int maxiter = vm["maxiter"].as<int>();
  double tol = vm["tolerance"].as<double>();

  hmm.TrainBaumWelch(seqs, maxiter, tol);

  hmm.SaveProfile(proout.c_str());

  return SUCCESS_PASS;
}

success_t train_viterbi_mixture() {
//  if (!fx_param_exists(NULL, "seqfile")) {
  if ( 0 == vm.count("seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }
  
  MixtureofGaussianHMM hmm;
  ArrayList<Matrix> seqs;

  //const char* seqin = fx_param_str_req(NULL, "seqfile");
  //const char* proout = fx_param_str(NULL, "profile", "pro.mix.out");

  std::string seqin = vm["seqfile"].as<std::string>();
  std::string proout = vm["profile"].as<std::string>();

  load_matrix_list(seqin.c_str(), &seqs);

  //if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
  //  const char* guess = fx_param_str_req(NULL, "guess");
  if ( 0 != vm.count("guess")) {
     if ( 0 == vm.count("guess")) {
       exit(1);
     }

     std::string guess = vm["guess"].as<std::string>();

    printf("Load parameters from file %s\n", guess.c_str());
    hmm.InitFromFile(guess.c_str());
  }
  else {
    hmm.Init();
    printf("Automatic initialization not supported !!!");
    return SUCCESS_FAIL;
  }

  //int maxiter = fx_param_int(NULL, "maxiter", 500);
  //double tol = fx_param_double(NULL, "tolerance", 1e-3);
  int maxiter = vm["maxiter"].as<int>();
  double tol = vm["tolerance"].as<double>();

  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout.c_str());

  return SUCCESS_PASS;
}

success_t train_viterbi_gaussian() {
//  if (!fx_param_exists(NULL, "seqfile")) {
  if ( 0 == vm.count("seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }
  
  GaussianHMM hmm;
  ArrayList<Matrix> seqs;

  //const char* seqin = fx_param_str_req(NULL, "seqfile");
  //const char* proout = fx_param_str(NULL, "profile", "pro.gauss.viterbi.out");

  std::string seqin = vm["seqfile"].as<std::string>();
  std::string proout = vm["profile"].as<std::string>();

  load_matrix_list(seqin.c_str(), &seqs);

 // if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
 //   const char* guess = fx_param_str_req(NULL, "guess");
  if ( 0 != vm.count("guess")) {
     if ( 0 == vm.count("guess")) {
       exit(1);
     }

     std::string guess = vm["guess"].as<std::string>();

    printf("Load parameters from file %s\n", guess.c_str());
    hmm.InitFromFile(guess.c_str());
  }
  else { // otherwise initialized using information from the data
    //int numstate = fx_param_int_req(NULL, "numstate");
    if ( 0 == vm.count("numstate")) {
       cerr << "Required parameter numstate not specified" << endl;
       exit(1);
    }
    int numstate = vm["numstate"].as<int>();

    printf("Generate parameters: NUMSTATE = %d\n", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  //int maxiter = fx_param_int(NULL, "maxiter", 500);
  //double tol = fx_param_double(NULL, "tolerance", 1e-3);
  int maxiter = vm["maxiter"].as<int>();
  double tol = vm["tolerance"].as<double>();

  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout.c_str());

  return SUCCESS_PASS;
}

success_t train_viterbi_discrete() {
  //if (!fx_param_exists(NULL, "seqfile")) {
 if ( 0 == vm.count("seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }

  DiscreteHMM hmm;
  ArrayList<Vector> seqs;

  //const char* seqin = fx_param_str_req(NULL, "seqfile");
  //const char* proout = fx_param_str(NULL, "profile", "pro.dis.viterbi.out");

  std::string seqin = vm["seqfile"].as<std::string>();
  std::string proout = vm["profile"].as<std::string>();

  load_vector_list(seqin.c_str(), &seqs);

 // if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
 //   const char* guess = fx_param_str_req(NULL, "guess");
   if ( 0 != vm.count("guess")) {
    ArrayList<Matrix> matlst;
     if ( 0 == vm.count("guess")) {
       exit(1);
     }

     std::string guess = vm["guess"].as<std::string>();
    printf("Load parameters from file %s\n", guess.c_str());
    hmm.InitFromFile(guess.c_str());
  }
  else { // otherwise randomly initialized using information from the data
    //int numstate = fx_param_int_req(NULL, "numstate");
    if ( 0 == vm.count("numstate")) {
       cerr << "Required parameter numstate not specified" << endl;
       exit(1);
    }
    int numstate = vm["numstate"].as<int>();

    printf("Generate parameters with NUMSTATE = %d\n", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  //int maxiter = fx_param_int(NULL, "maxiter", 500);
  //double tol = fx_param_double(NULL, "tolerance", 1e-3);
  int maxiter = vm["maxiter"].as<int>();
  double tol = vm["tolerance"].as<double>();

  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout.c_str());

  return SUCCESS_PASS;
}
