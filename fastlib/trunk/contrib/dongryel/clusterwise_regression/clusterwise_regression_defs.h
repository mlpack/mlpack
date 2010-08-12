/** @file clusterwise_regression_defs.h
 *
 *  @brief A parameter specification for clusterwise regression via
 *         the EM algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEFS_H
#define MLPACK_CLUSTERWISE_REGRESSION_CLUSTERWISE_REGRESSION_DEFS_H

#include <iostream>
#include "boost/program_options.hpp"
#include "boost/lexical_cast.hpp"
#include "fastlib/fastlib.h"
#include "clusterwise_regression.h"

int fl::ml::ClusterwiseRegression::Main(
  const std::vector<std::string> &args) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::vector<std::string> >(),
    "REQUIRED file containing reference data, if you want to run kda, you can "
    "provide more than one references files. Each reference file will contain files from the same class"
  )(
    "loglevel",
    boost::program_options::value<std::string>()->default_value("debug"),
    "Level of log detail.  One of:\n"
    "  debug: log everything\n"
    "  verbose: log messages and warnings\n"
    "  warning: log only warnings\n"
    "  silent: no logging"
  );

  boost::program_options::variables_map vm;
  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
  try {
    boost::program_options::store(clp.options(desc).run(), vm);
  }
  catch (const boost::program_options::invalid_option_value &e) {
    std::cerr << "Invalid Argument: " << e.what();
    throw new std::exception();
  }
  catch (const boost::program_options::invalid_command_line_syntax &e) {
    std::cerr << "Invalid command line syntax: " << e.what();
    throw new std::exception();
  }
  catch (const boost::program_options::unknown_option &e) {
    std::cerr << "Unknown option: " << e.what() ;
    throw new std::exception();
  }

  boost::program_options::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return true;
  }

  // Validate the  Only immediate dying is allowed here, the
  // parsing is done later.
  if (vm.count("references_in") == 0) {
    std::cerr << "Missing required --references_in";
    exit(-1);
  }

  return 0;
}

#endif
