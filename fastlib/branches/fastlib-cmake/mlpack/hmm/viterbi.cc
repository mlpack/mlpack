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
 * @file viterbi.cc
 *
 * This file contains the program to compute the most probable state sequences
 * in a Hidden Markov Model of given sequences 
 * Model.
 *
 * Usage:
 *   viterbi --type=TYPE --profile=PROFILE --seqfile=FILE [OPTIONS]
 * See the usage() function for complete option list
 */

#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace boost_po = boost::program_options;
boost_po::variables_map vm;

using namespace std;
using namespace hmm_support;

success_t viterbi_discrete();
success_t viterbi_gaussian();
success_t viterbi_mixture();
void usage();

const fx_entry_doc hmm_viterbi_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM profile.\n"},
  {"seqfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  {"statefile", FX_PARAM, FX_STR, NULL,
   "  Output file for the most probable state sequences.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_viterbi_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_viterbi_main_doc = {
  hmm_viterbi_main_entries, hmm_viterbi_main_submodules,
  "This is a program computing the most probable state sequences \n"
  "of data sequences from HMM models.\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_viterbi_main_doc);
  success_t s = SUCCESS_PASS;
  
  boost_po::options_description desc("Allowed options");
  desc.add_options()
      ("type", boost_po::value<std::string>(), "  HMM type : discrete | gaussian | mixture.\n")
      ("profile", boost_po::value<std::string>(), "  A file containing HMM profile.\n")
      ("seqfile", boost_po::value<std::string>(), "  Output file for the data sequences.\n")
      ("statefile", boost_po::value<std::string>(), "  Output file for the most probable state sequences.\n");

  boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
  boost_po::notify(vm);

//  if (fx_param_exists(NULL,"type")) {
   if ( 0 != vm.count("type")) {
    //const char* type = fx_param_str_req(NULL, "type");
    std::string type = vm["type"].as<std::string>();
    if (strcmp(type.c_str(), "discrete")==0)
      s = viterbi_discrete();
    else if (strcmp(type.c_str(), "gaussian")==0)
      s = viterbi_gaussian();
    else if (strcmp(type.c_str(), "mixture")==0)
      s = viterbi_mixture();
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
	 "  viterbi --type=={discrete|gaussian|mixture} OPTIONS\n"
	 "[OPTIONS]\n"
	 "  --profile=file   : file contains HMM profile\n"
	 "  --seqfile=file   : file contains input sequences\n"
	 "  --statefile=file : output file for state sequences\n"
	 );
}

success_t viterbi_mixture() {
//  if (!fx_param_exists(NULL, "profile")) {
  if ( 0 == vm.count("profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }

//  const char* profile = fx_param_str_req(NULL, "profile");
//  const char* seqin = fx_param_str(NULL, "seqfile", "seq.mix.out");
//  const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.mix.out");

  std::string profile = vm["profile"].as<std::string>();
  std::string seqin = vm["seqfile"].as<std::string>();
  std::string stateout = vm["statefile"].as<std::string>();

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile.c_str());

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin.c_str(), &seqs);

  TextWriter w_state;
  if (!PASSED(w_state.Open(stateout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout.c_str());
    return SUCCESS_FAIL;
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];

    hmm.ComputeViterbiStateSequence(seqs[i], &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return SUCCESS_PASS;
}

success_t viterbi_gaussian() {
  //if (!fx_param_exists(NULL, "profile")) {
  if ( 0 == vm.count("profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  //const char* profile = fx_param_str_req(NULL, "profile");
  //const char* seqin = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  //const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.gauss.out");
  
  std::string profile = vm["profile"].as<std::string>();
  std::string seqin = vm["seqfile"].as<std::string>();
  std::string stateout = vm["statefile"].as<std::string>();

  GaussianHMM hmm;
  hmm.InitFromFile(profile.c_str());

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin.c_str(), &seqs);

  TextWriter w_state;
  if (!PASSED(w_state.Open(stateout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout.c_str());
    return SUCCESS_FAIL;
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];
    hmm.ComputeViterbiStateSequence(seqs[i], &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return SUCCESS_PASS;
}

success_t viterbi_discrete() {
  //if (!fx_param_exists(NULL, "profile")) {
  if ( 0 == vm.count("profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  //const char* profile = fx_param_str_req(NULL, "profile");
  //const char* seqin = fx_param_str(NULL, "seqfile", "seq.out");
  //const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.out");

  std::string profile = vm["profile"].as<std::string>();
  std::string seqin = vm["seqfile"].as<std::string>();
  std::string stateout = vm["statefile"].as<std::string>();

  DiscreteHMM hmm;

  hmm.InitFromFile(profile.c_str());

  ArrayList<Vector> seqs;
  load_vector_list(seqin.c_str(), &seqs);

  TextWriter w_state;
  if (!PASSED(w_state.Open(stateout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout.c_str());
    return SUCCESS_FAIL;
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];
    
    hmm.ComputeViterbiStateSequence(seqs[i], &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return SUCCESS_PASS;
}
