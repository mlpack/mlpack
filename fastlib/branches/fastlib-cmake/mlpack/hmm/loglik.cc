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
 * @file loglik.cc
 *
 * This file contains the program to compute log-likelihood of sequences
 * according to a Hidden Markov  Model.
 *
 * Usage:
 *   loglik --type=TYPE --profile=PROFILE [OPTIONS]
 * See the usage() function for complete option list
 */

#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"

#include <boost/program_options.hpp>

namespace boost_po = boost::program_options;
boost_po::variables_map vm;

using namespace hmm_support;

success_t loglik_discrete();
success_t loglik_gaussian();
success_t loglik_mixture();
void usage();

const fx_entry_doc hmm_loglik_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM profile.\n"},
  {"seqfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  {"logfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the computed log-likelihood of the sequences.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_loglik_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_loglik_main_doc = {
  hmm_loglik_main_entries, hmm_loglik_main_submodules,
  "This is a program computing log-likelihood of data sequences \n"
  "from HMM models.\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_loglik_main_doc);
  success_t s = SUCCESS_PASS;
 
  boost_po::options_description desc("Allowed options");
  desc.add_options()
      ("type", boost_po::value<std::string>(), "  HMM type : discrete | gaussian | mixture.\n")
      ("profile", boost_po::value<std::string>(), "  A file containing HMM profile.\n")
      ("seqfile", boost_po::value<std::string>(), "  Output file for the data sequences.\n")
      ("logfile", boost_po::value<std::string>(), "  Output file for the computed log-likelihood of the sequences.\n");

  boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
  boost_po::notify(vm);

//  if (fx_param_exists(NULL,"type")) {
  if ( 0 != vm.count("type")) {
    //const char* type = fx_param_str_req(NULL, "type");
    if ( 0 == vm.count("type")) {
      exit(1);
    }
    std::string type = vm["type"].as<std::string>();
    if (strcmp(type.c_str(), "discrete")==0)
      s = loglik_discrete();
    else if (strcmp(type.c_str(), "gaussian")==0) 
      s = loglik_gaussian();
    else if (strcmp(type.c_str(), "mixture")==0) 
      s = loglik_mixture();
    else {
      printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
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

void usage() {
  printf("\n"
	 "Usage:\n"
	 "  loglik --type=={discrete|gaussian|mixture} OPTIONS\n"
	 "[OPTIONS]\n"
	 "  --profile==file   : file contains HMM profile\n"
	 "  --seqfile==file   : file contains input sequences\n"
	 "  --logfile==file   : output file for log-likelihood of the sequences\n"
	 );
}

success_t loglik_mixture() {
//  if (!fx_param_exists(NULL, "profile")) {
  if ( 0 == vm.count("profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  //const char* profile = fx_param_str_req(NULL, "profile");
  //const char* seqin = fx_param_str(NULL, "seqfile", "seq.mix.out");
  //const char* logout = fx_param_str(NULL, "logfile", "log.mix.out");

  if ( 0 == vm.count("profile")) {
    exit(1); 
  }
  std::string profile = vm["profile"].as<std::string>();
  std::string seqin = vm["seqfile"].as<std::string>();
  std::string logout = vm["logfile"].as<std::string>();

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile.c_str());

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin.c_str(), &seqs);

  TextWriter w_log;
  if (!PASSED(w_log.Open(logout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", logout.c_str());
    return SUCCESS_FAIL;
  }

  ArrayList<double> list_loglik;
  hmm.ComputeLogLikelihood(seqs, &list_loglik);

  for (int i = 0; i < seqs.size(); i++)
    w_log.Printf("%f\n", list_loglik[i]);
  
  return SUCCESS_PASS;
}

success_t loglik_gaussian() {
//  if (!fx_param_exists(NULL, "profile")) {
  if ( 0 == vm.count("profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  //const char* profile = fx_param_str_req(NULL, "profile");
  //const char* seqin = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  //const char* logout = fx_param_str(NULL, "logfile", "log.gauss.out");

  if ( 0 == vm.count("profile")) {
    exit(1);
  }
  std::string profile = vm["profile"].as<std::string>();
  std::string seqin = vm["seqfile"].as<std::string>();
  std::string logout = vm["logfile"].as<std::string>();

  GaussianHMM hmm;
  hmm.InitFromFile(profile.c_str());

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin.c_str(), &seqs);

  TextWriter w_log;
  if (!PASSED(w_log.Open(logout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", logout.c_str());
    return SUCCESS_FAIL;
  }

  ArrayList<double> list_loglik;
  hmm.ComputeLogLikelihood(seqs, &list_loglik);

  for (int i = 0; i < seqs.size(); i++)
    w_log.Printf("%f\n", list_loglik[i]);
  
  return SUCCESS_PASS;
}

success_t loglik_discrete() {
//  if (!fx_param_exists(NULL, "profile")) {
  if ( 0 == vm.count("profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
//  const char* profile = fx_param_str_req(NULL, "profile");
//  const char* seqin = fx_param_str(NULL, "seqfile", "seq.out");
//  const char* logout = fx_param_str(NULL, "logfile", "log.out");

  if ( 0 == vm.count("profile")) {
    exit(1);
  }
  std::string profile = vm["profile"].as<std::string>();
  std::string seqin = vm["seqfile"].as<std::string>();
  std::string logout = vm["logfile"].as<std::string>();

  DiscreteHMM hmm;
  hmm.InitFromFile(profile.c_str());

  ArrayList<Vector> seqs;
  load_vector_list(seqin.c_str(), &seqs);

  TextWriter w_log;
  if (!PASSED(w_log.Open(logout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", logout.c_str());
    return SUCCESS_FAIL;
  }

  ArrayList<double> list_loglik;
  hmm.ComputeLogLikelihood(seqs, &list_loglik);

  for (int i = 0; i < seqs.size(); i++)
    w_log.Printf("%f\n", list_loglik[i]);
  return SUCCESS_PASS;
}

