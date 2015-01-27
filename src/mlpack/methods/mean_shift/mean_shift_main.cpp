/**
 * @file mean_shift_main.cpp
 * @author Shangtong Zhang
 *
 * Executable for running Mean Shift
 */

#include <mlpack/core.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include "mean_shift.hpp"

using namespace mlpack;
using namespace mlpack::meanshift;
using namespace std;

// Define parameters for the executable.
PROGRAM_INFO("Mean Shift Clustering", "This program performs mean shift clustering "
             "on the given dataset, storing the learned cluster assignments either as "
             "a column of labels in the file containing the input dataset or in a "
             "separate file. "
             "\n\n");

// Required options.
PARAM_STRING_REQ("inputFile", "Input dataset to perform clustering on.", "i");

// Output options.
PARAM_FLAG("in_place", "If specified, a column containing the learned cluster "
           "assignments will be added to the input dataset file.  In this case, "
           "--outputFile is overridden.", "P");
PARAM_STRING("output_file", "File to write output labels or labeled data to.",
             "o", "");
PARAM_STRING("centroid_file", "If specified, the centroids of each cluster will"
             " be written to the given file.", "C", "");

// Mean Shift configuration options.
PARAM_INT("max_iterations", "Maximum number of iterations before Mean Shift "
          "terminates.", "m", 1000);
PARAM_DOUBLE("stopThresh", "If the 2-norm of the mean shift vector "
             "is less than stopThresh, iterations will terminate. ", "s", 1e-3);
PARAM_DOUBLE("bandwidth", "bandwidth of Gaussian kernel ", "b", 1.0);
PARAM_DOUBLE("duplicateThresh", "If distance of two centroids is less than duplicate thresh "
             "one will be removed. "
             "If it's negative, an estimation will be given. ", "d", -1.0);


int main(int argc, char** argv) {
  
  CLI::ParseCommandLine(argc, argv);
  
  const string inputFile = CLI::GetParam<string>("inputFile");
  const double stopThresh = CLI::GetParam<double>("stopThresh");
  const double bandwidth = CLI::GetParam<double>("bandwidth");
  const int maxIterations = CLI::GetParam<int>("max_iterations");
  const double duplicateThresh = CLI::GetParam<double>("duplicateThresh");
  
  if (maxIterations < 0) {
    Log::Fatal << "Invalid value for maximum iterations (" << maxIterations <<
    ")! Must be greater than or equal to 0." << endl;
  }
  
  // Make sure we have an output file if we're not doing the work in-place.
  if (!CLI::HasParam("in_place") && !CLI::HasParam("output_file") &&
      !CLI::HasParam("centroid_file")) {
    Log::Warn << "--output_file, --in_place, and --centroid_file are not set; "
    << "no results will be saved." << std::endl;
  }
  
  arma::mat dataset;
  data::Load(inputFile, dataset, true); // Fatal upon failure.
  arma::mat centroids;
  arma::Col<size_t> assignments;
  
  kernel::GaussianKernel kernel;
  kernel.Bandwidth(bandwidth);
  
  MeanShift<> meanShift(duplicateThresh,
                                                         maxIterations, stopThresh, kernel);
  Timer::Start("clustering");
  meanShift.Cluster(dataset, assignments, centroids);
  Timer::Stop("clustering");
  
  if (CLI::HasParam("in_place")) {
    // Add the column of assignments to the dataset; but we have to convert
    // them to type double first.
    arma::vec converted(assignments.n_elem);
    for (size_t i = 0; i < assignments.n_elem; i++)
      converted(i) = (double) assignments(i);
    
    dataset.insert_rows(dataset.n_rows, trans(converted));
    
    // Save the dataset.
    data::Save(inputFile, dataset);
  } else {
    
    // Convert the assignments to doubles.
    arma::vec converted(assignments.n_elem);
    for (size_t i = 0; i < assignments.n_elem; i++)
      converted(i) = (double) assignments(i);
    
    dataset.insert_rows(dataset.n_rows, trans(converted));
    
    // Now save, in the different file.
    string outputFile = CLI::GetParam<string>("output_file");
    data::Save(outputFile, dataset);

  }
  
  // Should we write the centroids to a file?
  if (CLI::HasParam("centroid_file")) {
    data::Save(CLI::GetParam<std::string>("centroid_file"), centroids);
  }
  
  
}