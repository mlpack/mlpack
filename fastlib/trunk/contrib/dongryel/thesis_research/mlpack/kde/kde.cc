/** @file kde.cc
 *
 *  The main driver for the KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <iostream>
#include <string>
#include <armadillo>
#include "kde_dev.h"

int main(int argc, char *argv[]) {

  // Parse arguments for Kde.
  ml::KdeArguments kde_arguments;
  ml::Kde::ParseArguments(argc, argv, &kde_arguments);

  // Instantiate a KDE object.
  ml::Kde kde_instance;
  kde_instance.Init(kde_arguments);

  // Compute the result.
  ml::KdeResult< std::vector<double> > kde_result;
  kde_instance.Compute(kde_arguments, &kde_result);

  // Output the KDE result to the file.
  std::cerr << "Writing the densities to the file: " <<
            kde_arguments.densities_out_ << "\n";
  kde_result.PrintDebug(kde_arguments.densities_out_);

  return 0;
}
