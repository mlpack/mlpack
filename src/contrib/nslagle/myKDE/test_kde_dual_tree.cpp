#include <mlpack/core.h>
#include "kde_dual_tree.hpp"

PROGRAM_INFO("Kernel Density Estimation Multibandwidth Dual Tree",
    "KDE multibandwidth dual tree calculates density estimates for each "
    "query point, given a collection of reference points, using a collection "
    "of equidistantly spaced bandwidths\n\n"
    "$ kde_dual_tree --reference_file=reference.csv --query_file=query.csv\n"
    "  --output_file=output.csv --low_bandwidth=0.1 --high_bandwidth=100.0\n"
    "  --bandwidth_count=10 --epsilon=0.01 --delta=0.01", "kde_dual_tree");

PARAM_STRING_REQ("reference_file", "CSV file containing the reference dataset.",
    "");
PARAM_STRING("query_file", "CSV file containing query points",
    "", "");
PARAM_STRING("output_file", "File to output CSV-formatted results into.", "",
    "kde_dual_tree_output.csv");
PARAM_DOUBLE("low_bandwidth", "Low bandwidth", "", 0.1);
PARAM_DOUBLE("high_bandwidth", "Low bandwidth", "", 100.0);
PARAM_INT("bandwidth_count", "Low bandwidth", "", 10);
PARAM_DOUBLE("epsilon", "error tolerance", "", 0.01);
PARAM_DOUBLE("delta", "reversibility tolerance", "", 0.01);

int main (int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);
  std::string referenceFile = CLI::GetParam<std::string>("reference_file");
  std::string queryFile = CLI::GetParam<std::string>("query_file");
  std::string outputFile = CLI::GetParam<std::string>("output_file");
  arma::mat referenceData;
  arma::mat queryData;
  double epsilon = CLI::GetParam<double>("epsilon");
  double delta = CLI::GetParam<double>("delta");
  int bandwidthCount = CLI::GetParam<int>("bandwidth_count");
  double lowBandwidth = CLI::GetParam<double>("low_bandwidth");
  double highBandwidth = CLI::GetParam<double>("high_bandwidth");

  /* check the parameters */
  if (delta < 0.0)
  {
    Log::Fatal << "Improper delta: " << delta <<
                  "; delta must be positive" << std::endl;
  }
  if (epsilon < 0.0)
  {
    Log::Fatal << "Improper epsilon: " << epsilon <<
                  "; epsilon must be positive" << std::endl;
  }
  if (bandwidthCount <= 0)
  {
    Log::Fatal << "Improper bandwidth_count: " << bandwidthCount <<
                  "; bandwidth_count must be positive" << std::endl;
  }
  if (highBandwidth <= lowBandwidth + DBL_EPSILON || lowBandwidth <= 0.0)
  {
    Log::Fatal << "Improper bandwidth range: " << lowBandwidth << ", " <<
                   highBandwidth << "; bandwidth range must be a positive interval" << std::endl;
  }

  if (!data::Load(referenceFile.c_str(), referenceData))
  {
    Log::Fatal << "Failed to load the reference file " << referenceFile << std::endl;
  }

  Log::Info << "Loaded reference data from " << referenceFile << std::endl;

  std::vector<double> densities;
  if (queryFile == "")
  {
    /* invoke KDE without specific query data */
    KdeDualTree<> kde = KdeDualTree<>(referenceData);
    kde.Epsilon() = epsilon;
    kde.Delta() = delta;
    kde.BandwidthCount() = bandwidthCount;
    kde.SetBandwidthBounds(lowBandwidth, highBandwidth);
    densities = kde.Calculate();
  }
  else
  {
    /* invoke KDE without specific query data */
    KdeDualTree<> kde = KdeDualTree<>(referenceData, queryData);
    kde.Epsilon() = epsilon;
    kde.Delta() = delta;
    kde.BandwidthCount() = bandwidthCount;
    kde.SetBandwidthBounds(lowBandwidth, highBandwidth);
    densities = kde.Calculate();
  }
  size_t index = 0;
  for (std::vector<double>::iterator dIt = densities.begin();
      dIt != densities.end();
      ++dIt)
  {
    if (*dIt != 0.0)
    {
      std::cout << "density[" << index << "]=" << *dIt << std::endl;
    }
    ++index;
  }
  return 0;
}
