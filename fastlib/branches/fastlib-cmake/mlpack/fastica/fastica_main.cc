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
 * @file fastica_main.cc
 *
 * Demonstrates usage of fastica.h
 *
 * @see fastica.h
 *
 * @author Nishant Mehta
 */
#include "fastica.h"

/**
 * Here are usage instructions for this implementation of FastICA. Default values are given 
 * to the right in parentheses.
 *
 * Parameters specific to this driver:
 *
 *   @param data = data file with each row being one sample (REQUIRED PARAMETER)
 *   @param ic_filename = independent components results filename (ic.dat)
 *   @param unmixing_filename = unmixing matrix results filename (unmixing.dat)
 *
 * Parameters specific to fastica.h (should be preceded by 'fastica/' on the command line):
 *
 *   @param seed = (long) seed to the random number generator (clock() + time(0))
 *   @param approach = {deflation, symmetric} (deflation)
 *   @param nonlinearity = {logcosh, gauss, kurtosis, skew} (logcosh)
 *   @param fine_tune = {true, false} (false)
 *   @param a1 = numeric constant for logcosh nonlinearity (1)
 *   @param a2 = numeric constant for gauss nonlinearity (1)
 *   @param mu = numeric constant for fine-tuning Newton-Raphson method (1)
 *   @param stabilization = {true, false} (false)
 *   @param epsilon = threshold for convergence (0.0001)
 *   @param max_num_iterations = maximum number of fixed-point iterations
 *   @param max_fine_tune = maximum number of fine-tuning iterations
 *   @param percent_cut = number in [0,1] indicating percent data to use in stabilization updates (1)
 *   
 * Example use:
 *
 * @code
 * ./fastica --data=X_t.dat --ic_filename=ic.dat --unmixing_filename=W.dat
 * --fastica/approach=symmetric --fastica/nonlinearity=gauss
 * --fastica/stabilization=true --fastica/epsilon=0.0000001 --percent_cut=0.5
 * @endcode
 *
 * Note: Compile with verbose mode to display convergence-related values
 */

const fx_entry_doc fastica_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   "  A file containing data.\n"},
  {"ic_filename", FX_PARAM, FX_STR, NULL,
   "  Filename to which independent components are written.\n"},
  {"unmixing_filename", FX_PARAM, FX_STR, NULL,
   "  Filename to which unmixing matrix is written.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc fastica_main_submodules[] = {
  {"fastica", &fastica_doc,
   " Responsible for performing fastica.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc fastica_main_doc = {
  fastica_main_entries, fastica_main_submodules,
  "This program performs fastica.\n"
};



int main(int argc, char *argv[]) {
  fx_module* root = fx_init(argc, argv, &fastica_main_doc);

  Matrix X;
  
  std::string data;
  std::string ic_filename;
  std::string unmixing_filename;

  boost_po::options_description desc("Allowed options");
  desc.add_options()
      ("data", boost_po::value<std::string>(&data), "  The dataset file name")
      ("ic_filename", boost_po::value<std::string>(&ic_filename)->default_value("ic.dat"), "  The dataset file name")
      ("unmixing_filename", boost_po::value<std::string>(&unmixing_filename)->default_value("unmixing.dat"), "  Unmixing file name")
      ("seed", boost_po::value<int>()->default_value(clock() + time(0)), "  Seed for the random number generator")
      ("approach", boost_po::value<std::string>()->default_value("deflation"), "  Independent component recovery approach: 'deflation' or 'symmetric'.\n")
      ("nonlinearity", boost_po::value<std::string>()->default_value("logcosh"), "  Nonlinear function to use: 'logcosh', 'gauss', 'kurtosis', or 'skew'.\n")
      ("fine_tune", boost_po::value<bool>()->default_value(false), "  Enable fine tuning.\n")
      ("a1", boost_po::value<double>()->default_value(1), "  Numeric constant for logcosh nonlinearity.\n")
      ("a2", boost_po::value<double>()->default_value(1), "  Numeric constant for gauss nonlinearity.\n")
      ("mu", boost_po::value<double>()->default_value(1), "  Numeric constant for fine-tuning Newton-Raphson method.\n")
      ("stabilization", boost_po::value<bool>()->default_value(false), "  Use stabilization.\n")
      ("epsilon", boost_po::value<double>()->default_value(0.0001), "  Threshold for convergence.\n")
      ("max_num_iterations", boost_po::value<int>()->default_value(1000), "  Maximum number of iterations of fixed-point iterations.\n")
      ("max_fine_tune", boost_po::value<int>()->default_value(5), "  Maximum number of iterations of fixed-point iterations.\n")
      ("percent_cut", boost_po::value<double>()->default_value(1), "  Maximum number of fine-tuning iterations.\n");

  boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
  boost_po::notify(vm);

  //const char* data = fx_param_str_req(NULL, "data");
  data::Load(data.c_str(), &X);

  //const char* ic_filename = fx_param_str(NULL, "ic_filename", "ic.dat");
  //const char* unmixing_filename =
  //  fx_param_str(NULL, "unmixing_filename", "unmixing.dat");
  struct datanode* fastica_module =
    fx_submodule(root, "fastica");

  FastICA fastica;
  
  int success_status = SUCCESS_FAIL;
  if(fastica.Init(X, fastica_module) == SUCCESS_PASS) {
    Matrix W, Y;
    if(fastica.DoFastICA(&W, &Y) == SUCCESS_PASS) {
      SaveCorrectly(unmixing_filename.c_str(), W);
      data::Save(ic_filename.c_str(), Y);
      success_status = SUCCESS_PASS;
      VERBOSE_ONLY( W.PrintDebug("W") );
    }
  }
  

  if(success_status == SUCCESS_FAIL) {
    VERBOSE_ONLY( printf("FAILED!\n") );
  }

  fx_done(root);

  return success_status;
}
