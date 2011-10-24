/** @file mixed_logit.test.cc
 *
 *  A "stress" test driver for mixed logit.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <numeric>
#include <time.h>
#include "core/table/table.h"
#include "core/tree/gen_metric_tree.h"
#include "mlpack/mixed_logit_dcm/constant_distribution.h"
#include "mlpack/mixed_logit_dcm/distribution.h"
#include "mlpack/mixed_logit_dcm/gaussian_distribution.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_argument_parser.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_dev.h"

namespace mlpack {
namespace mixed_logit_dcm {

int num_attributes_;
int num_people_;
std::vector<int> attribute_dimensions_;

class TestMixedLogitDCM {

  private:

    void InitializeAttributeDimensions_(
      const std::string &distribution_name) {

      if(distribution_name == "constant") {
        mlpack::mixed_logit_dcm::attribute_dimensions_.resize(1);
        mlpack::mixed_logit_dcm::attribute_dimensions_[0] =
          mlpack::mixed_logit_dcm::num_attributes_;
      }
      else if(distribution_name == "diag_gaussian") {
        mlpack::mixed_logit_dcm::attribute_dimensions_.resize(1);
        mlpack::mixed_logit_dcm::attribute_dimensions_[0] =
          mlpack::mixed_logit_dcm::num_attributes_;
      }
      else if(distribution_name == "full_gaussian") {
        mlpack::mixed_logit_dcm::attribute_dimensions_.resize(3);
        mlpack::mixed_logit_dcm::attribute_dimensions_[0] =
          mlpack::mixed_logit_dcm::num_attributes_ / 3;
        mlpack::mixed_logit_dcm::attribute_dimensions_[1] =
          mlpack::mixed_logit_dcm::attribute_dimensions_[0];
        mlpack::mixed_logit_dcm::attribute_dimensions_[2] =
          mlpack::mixed_logit_dcm::num_attributes_ -
          mlpack::mixed_logit_dcm::attribute_dimensions_[0] -
          mlpack::mixed_logit_dcm::attribute_dimensions_[1];
      }
    }

    template<typename TableType>
    void GenerateRandomDataset_(
      const std::string &distribution_name,
      TableType *random_attribute_dataset,
      TableType *random_decisions_dataset,
      TableType *random_num_alternatives_dataset) {


    }

  public:

    int StressTestMain() {
      for(int i = 0; i < 1; i++) {
        for(int k = 0; k < 3; k++) {

          // Randomly choose the number of attributes and the number
          // of people and the number of discrete choices per each
          // person.
          mlpack::mixed_logit_dcm::num_attributes_ =
            core::math::RandInt(10, 20);
          mlpack::mixed_logit_dcm::num_people_ = core::math::RandInt(50, 70);

          switch(k) {
            case 0:

              // Test the constant distribution.
              StressTest <
              mlpack::mixed_logit_dcm::ConstantDistribution > (
                std::string("constant"));
              break;
            case 1:

              // Test the diagonal Gaussian distribution.
              StressTest <
              mlpack::mixed_logit_dcm::DiagonalGaussianDistribution > (
                std::string("diag_gaussian"));
              break;
            case 2:

              // Test the full Gaussian distribution.
              StressTest <
              mlpack::mixed_logit_dcm::GaussianDistribution > (
                std::string("full_gaussian"));
              break;
          }
        }
      }
      return 0;
    }

    template<typename DistributionType>
    int StressTest(const std::string &distribution_name) {

      typedef core::table::Table <
      core::tree::GenMetricTree <
      core::tree::AbstractStatistic > ,
           mlpack::mixed_logit_dcm::MixedLogitDCMResult > TableType;

      // The list of arguments.
      std::vector< std::string > args;

      // Push in the distribution type.
      args.push_back(std::string("--distribution_in=") + distribution_name);

      // Push in the reference dataset name.
      std::string attributes_in("random_attributes.csv");
      args.push_back(std::string("--attributes_in=") + attributes_in);

      // Push in the decision set info name.
      std::string decisions_in(
        "random_decisions.csv");
      args.push_back(
        std::string("--decisions_in=") +
        decisions_in);

      // Push in the number of alternatives.
      std::string num_alternatives_in("random_num_alternatives.csv");
      args.push_back(
        std::string("--num_alternatives_in=") +
        num_alternatives_in);

      // Print out the header of the trial.
      std::cout << "\n==================\n";
      std::cout << "Test trial begin\n";
      std::cout << "Number of attributes: " <<
                mlpack::mixed_logit_dcm::num_attributes_ << "\n";
      std::cout << "Number of people: " <<
                mlpack::mixed_logit_dcm::num_people_ << "\n";

      // Generate the random dataset and save it.
      mlpack::mixed_logit_dcm::DCMTable <
      TableType, DistributionType > random_dcm_table;
      InitializeAttributeDimensions_(distribution_name);
      random_dcm_table.GenerateRandomDataset(
        mlpack::mixed_logit_dcm::num_people_,
        mlpack::mixed_logit_dcm::num_attributes_,
        mlpack::mixed_logit_dcm::attribute_dimensions_);
      random_dcm_table.Save(attributes_in, decisions_in, num_alternatives_in);

      // Parse the mixed logit DCM arguments.
      mlpack::mixed_logit_dcm::MixedLogitDCMArguments<TableType> arguments;
      boost::program_options::variables_map vm;
      mlpack::mixed_logit_dcm::MixedLogitDCMArgumentParser::
      ConstructBoostVariableMap(args, &vm);
      mlpack::mixed_logit_dcm::MixedLogitDCMArgumentParser::ParseArguments(
        vm, &arguments);

      // Call the mixed logit driver.
      mlpack::mixed_logit_dcm::MixedLogitDCM <
      TableType, DistributionType > instance;
      instance.Init(arguments);

      // Compute the result.
      mlpack::mixed_logit_dcm::MixedLogitDCMResult result;
      instance.Train(arguments, &result);

      return 0;
    };
};
}
}

int main(int argc, char *argv[]) {

  // Call the tests.
  mlpack::mixed_logit_dcm::TestMixedLogitDCM dcm_test;
  dcm_test.StressTestMain();

  std::cout << "All tests passed!\n";
  return 0;
}
