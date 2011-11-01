/**
* @file emst.cc
 *
 * Calls the DualTreeBoruvka algorithm from dtb.h
 * Can optionally call Naive Boruvka's method
 * See README for command line options.
 *
 * @author Bill March (march@gatech.edu)
*/


#include <mlpack/core.h>
#include "dtb.hpp"

PARAM_FLAG("using_thor", "For when an implementation of thor is around",
    "emst");
PARAM_STRING_REQ("input_file", "Data input file.", "emst");
PARAM_STRING("output_file", "Data output file.", "emst", "emst_output.csv");

PARAM_FLAG("do_naive", "Check against naive.", "naive");
PARAM_STRING("output_file", "Naive data output file.", "naive",
    "naive_output.csv");

PARAM(double, "total_squared_length", "Calculation result.", "dtb", 0.0, false);

using namespace mlpack;
using namespace mlpack::emst;

int main(int argc, char* argv[]) {
  CLI::ParseCommandLine(argc, argv);

  // For when I implement a thor version
  bool using_thor = CLI::GetParam<bool>("emst/using_thor");


  if (using_thor) {
    Log::Warn << "thor is not yet supported" << std::endl;
  }
  else {

    ///////////////// READ IN DATA //////////////////////////////////

    std::string data_file_name = CLI::GetParam<std::string>("emst/input_file");

    arma::mat data_points;
    data_points.load(data_file_name.c_str());

    /////////////// Initialize DTB //////////////////////
    DualTreeBoruvka dtb;

    ////////////// Run DTB /////////////////////
    arma::mat results;

    dtb.ComputeMST(results);

    //////////////// Check against naive //////////////////////////
    if (CLI::GetParam<bool>("naive/do_naive")) {

      DualTreeBoruvka naive;
      CLI::GetParam<bool>("naive/do_naive") = true;

      naive.Init(data_points);

      arma::mat naive_results;
      naive.ComputeMST(naive_results);

      /* Compare the naive output to the DTB output */

      Timers::StartTimer("naive/comparison");


      // Check if the edge lists are the same
      // Loop over the naive edge list
      int is_correct = 1;
      /*
      for (size_t naive_index = 0; naive_index < results.size();
           naive_index++) {

        int this_loop_correct = 0;
        size_t naive_lesser_index = results[naive_index].lesser_index();
        size_t naive_greater_index = results[naive_index].greater_index();
        double naive_distance = results[naive_index].distance();

        // Loop over the DTB edge list and compare against naive
        // Break when an edge is found that matches the current naive edge
        for (size_t dual_index = 0; dual_index < naive_results.size();
             dual_index++) {

          size_t dual_lesser_index = results[dual_index].lesser_index();
          size_t dual_greater_index = results[dual_index].greater_index();
          double dual_distance = results[dual_index].distance();

          if (naive_lesser_index == dual_lesser_index) {
            if (naive_greater_index == dual_greater_index) {
              DEBUG_ASSERT(naive_distance == dual_distance);
              this_loop_correct = 1;
              break;
            }
          }

        }

        if (this_loop_correct == 0) {
          is_correct = 0;
          break;
        }

      }
     */
      if (is_correct == 0) {

        Log::Warn << "Naive check failed!" << std::endl <<
        "Edge lists are different." << std::endl << std::endl;

        // Check if the outputs have the same length
        if (CLI::GetParam<double>("naive/total_squared_length") !=
           CLI::GetParam<double>("naive/total_squared_length")) {

          Log::Fatal << "Total lengths are different! "
             << " One algorithm has failed." << std::endl;

          return 1;
        }
        else {
          // NOTE: if the edge lists are different, but the total lengths are
          // the same, the algorithm may still be correct.  The MST is not
          // uniquely defined for some point sets.  For example, an equilateral
          // triangle has three minimum spanning trees.  It is possible for
          // naive and DTB to find different spanning trees in this case.
          Log::Info << "Total lengths are the same.";
          Log::Info << "It is possible the point set";
          Log::Info << "has more than one minimum spanning tree." << std::endl;
        }

      }
      else {
        Log::Info << "Naive and DualTreeBoruvka produced the same MST." <<
          std::endl << std::endl;
      }

      Timers::StopTimer("naive/comparison");

      std::string naive_output_filename =
          CLI::GetParam<std::string>("naive/output_file");

      naive_results.save(naive_output_filename.c_str(), arma::csv_ascii, false,
          true);
    }

    //////////////// Output the Results ////////////////

    std::string output_filename =
        CLI::GetParam<std::string>("emst/output_file");

    results.save(output_filename.c_str(), arma::csv_ascii, false, true);

  }// end else (if using_thor)

  return 0;

}
