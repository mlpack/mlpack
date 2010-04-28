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
 * @file generate.cc
 *
 * This file contains the program to generate sequences from a Hidden Markov
 * Model.
 *
 * Usage:
 *   generate --type=TYPE --profile=PROFILE [OPTIONS]
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

success_t generate_discrete();
success_t generate_gaussian();
success_t generate_mixture();
void usage();

const fx_entry_doc hmm_generate_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM profile.\n"},
  {"length", FX_PARAM, FX_INT, NULL,
   "  Sequence length, default = 10.\n"},
  {"lenmax", FX_PARAM, FX_INT, NULL,
   "  Maximum sequence length, default = length\n"},
  {"numseq", FX_PARAM, FX_INT, NULL,
   "  Number of sequance, default = 10.\n"},
  {"seqfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the generated sequences.\n"},
  {"statefile", FX_PARAM, FX_STR, NULL,
   "  Output file for the generated state sequences.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_generate_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_generate_main_doc = {
  hmm_generate_main_entries, hmm_generate_main_submodules,
  "This is a program generating sequences from HMM models.\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_generate_main_doc );
  success_t s = SUCCESS_PASS;
 
  boost_po::options_description desc("Allowed options");
  desc.add_options()
      ("type", boost_po::value<std::string>(), "  HMM type : discrete | gaussian | mixture.\n")
      ("profile", boost_po::value<std::string>(), "  A file containing HMM profile.\n")
      ("length", boost_po::value<int>()->default_value(10), "  Sequence length, default = 10.\n")
      ("lenmax", boost_po::value<int>(), "  Maximum sequence length, default = length\n")
      ("numseq", boost_po::value<int>()->default_value(10), "  Number of sequance, default = 10.\n")
      ("seqfile", boost_po::value<std::string>()->default_value("seq.mix.out"), "  Output file for the generated sequences.\n")
      ("statefile", boost_po::value<std::string>()->default_value("state.mix.out"), "  Output file for the generated state sequences.\n");

  boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
  boost_po::notify(vm);

//  if (fx_param_exists(NULL,"type")) {
  if ( 0 != vm.count("type")) {
    //const char* type = fx_param_str_req(NULL, "type");
    if ( 0 == vm.count("type") ) {
      cerr << "Required parameter type not entered" << endl;
      exit(1);
    }
    std::string type = vm["type"].as<std::string>();

    if (strcmp(type.c_str(), "discrete")==0)
      s = generate_discrete();
    else if (strcmp(type.c_str(), "gaussian")==0) 
      s = generate_gaussian();
    else if (strcmp(type.c_str(), "mixture")==0)
      s = generate_mixture();
    else {
      printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
      return SUCCESS_PASS;
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
  printf("\nUsage:\n");
  printf("  generate --type=={discrete|gaussian|mixture} OPTIONS\n");
  printf("[OPTIONS]\n");
  printf("  --profile=file   : file contains HMM profile\n");
  printf("  --length=NUM     : sequence length\n");
  printf("  --lenmax=NUM     : maximum sequence length, default = length\n");
  printf("  --numseq=NUM     : number of sequence\n");
  printf("  --seqfile=file   : output file for generated sequences\n");
  printf("  --statefile=file : output file for generated state sequences\n");
}

success_t generate_mixture() {
//  if (!fx_param_exists(NULL, "profile")) {
  if ( 0 == vm.count("profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
/*  const char* profile = fx_param_str_req(NULL, "profile");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "seqfile", "seq.mix.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.mix.out");
*/

  std::string profile = vm["profile"].as<std::string>();
  const int seqlen = vm["length"].as<int>();
  int seqlmax;
  const int numseq = vm["numseq"].as<int>();
  std::string seqout = vm["seqfile"].as<std::string>();
  std::string stateout = vm["statefile"].as<std::string>();

  if ( 0 ==  vm.count("lenmax")) {
    seqlmax = seqlen;
  }
  else {
    seqlmax = vm["lenmax"].as<int>();
  }

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax-seqlen) / numseq;

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile.c_str());
  
  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", seqout.c_str());
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout.c_str());
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    Matrix seq;
    Vector states;
    char s[100];

    hmm.GenerateSequence((int)L, &seq, &states);
    
    sprintf(s, "%% sequence %d", i);
    print_matrix(w_seq, seq, s, "%E,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }

  //printf("---END---");
  return SUCCESS_PASS;
}

success_t generate_gaussian() {
//  if (!fx_param_exists(NULL, "profile")) {
  if ( 0 == vm.count("profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  /*const char* profile = fx_param_str_req(NULL, "profile");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.gauss.out");
  */

  std::string profile = vm["profile"].as<std::string>();
  const int seqlen = vm["length"].as<int>();
  int seqlmax;
  const int numseq = vm["numseq"].as<int>();
  std::string seqout = vm["seqfile"].as<std::string>();
  std::string stateout = vm["statefile"].as<std::string>();

  if ( 0 ==  vm.count("lenmax")) {
    seqlmax = seqlen;
  }
  else {
    seqlmax = vm["lenmax"].as<int>();
  }

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax-seqlen) / numseq;

  GaussianHMM hmm;
  hmm.InitFromFile(profile.c_str());
  
  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", seqout.c_str());
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout.c_str());
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    Matrix seq;
    Vector states;
    char s[100];

    hmm.GenerateSequence((int)L, &seq, &states);
    
    sprintf(s, "%% sequence %d", i);
    print_matrix(w_seq, seq, s, "%E,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }
  return SUCCESS_PASS;
}

success_t generate_discrete() {
//  if (!fx_param_exists(NULL, "profile")) {
  if ( 0 == vm.count("profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
/*  const char* profile = fx_param_str_req(NULL, "profile");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "seqfile", "seq.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.out");
*/

  std::string profile = vm["profile"].as<std::string>();
  const int seqlen = vm["length"].as<int>();
  int seqlmax;
  const int numseq = vm["numseq"].as<int>();
  std::string seqout = vm["seqfile"].as<std::string>();
  std::string stateout = vm["statefile"].as<std::string>();

  if ( 0 ==  vm.count("lenmax")) {
    seqlmax = seqlen;
  }
  else {
    seqlmax = vm["lenmax"].as<int>();
  }

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax-seqlen) / numseq;

  DiscreteHMM hmm;
  hmm.InitFromFile(profile.c_str());

  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", seqout.c_str());
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout.c_str()))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout.c_str());
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    Vector seq, states;
    char s[100];

    hmm.GenerateSequence((int)L, &seq, &states);
    
    sprintf(s, "%% sequence %d", i);
    print_vector(w_seq, seq, s, "%.0f,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }
  return SUCCESS_PASS;
}
