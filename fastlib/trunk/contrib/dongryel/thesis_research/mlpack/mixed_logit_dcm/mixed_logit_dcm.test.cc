/** @file mixed_logit.test.cc
 *
 *  A "stress" test driver for mixed logit.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <numeric>
#include <time.h>
#include "core/table/table.h"
#include "core/tree/gen_metric_tree.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_dev.h"

namespace mlpack {
namespace mixed_logit_dcm {

int num_attributes_;
int num_people_;
std::vector<int> num_discrete_choices_;

class TestMixedLogitDCM {

  private:

    template<typename TableType>
    void GenerateRandomDataset_(
      TableType *random_attribute_dataset,
      TableType *random_discrete_choice_set_info_dataset) {

      // Find the total number of discrete choices.
      int total_num_attributes =
        std::accumulate(
          mlpack::mixed_logit_dcm::num_discrete_choices_.begin(),
          mlpack::mixed_logit_dcm::num_discrete_choices_.end(), 0);

      random_attribute_dataset->Init(
        mlpack::mixed_logit_dcm::num_attributes_, total_num_attributes);

      for(int j = 0; j < total_num_attributes; j++) {
        core::table::DensePoint point;
        random_attribute_dataset->get(j, &point);
        for(int i = 0; i < mlpack::mixed_logit_dcm::num_attributes_; i++) {
          point[i] = core::math::Random(0.1, 1.0);
        }
      }

      random_discrete_choice_set_info_dataset->Init(
        2, mlpack::mixed_logit_dcm::num_people_);
      for(int j = 0; j < mlpack::mixed_logit_dcm::num_people_; j++) {
        core::table::DensePoint point;
        random_discrete_choice_set_info_dataset->get(j, &point);
        point[0] = core::math::RandInt(
                     mlpack::mixed_logit_dcm::num_discrete_choices_[j]);
        point[1] = mlpack::mixed_logit_dcm::num_discrete_choices_[j];
      }
    }

  public:

    int StressTestMain() {
      for(int i = 0; i < 20; i++) {
        for(int k = 0; k < 2; k++) {

          // Randomly choose the number of attributes and the number
          // of people and the number of discrete choices per each
          // person.
          mlpack::mixed_logit_dcm::num_attributes_ = core::math::RandInt(5, 20);
          mlpack::mixed_logit_dcm::num_people_ = core::math::RandInt(50, 101);
          mlpack::mixed_logit_dcm::num_discrete_choices_.resize(
            mlpack::mixed_logit_dcm::num_people_);

          switch(k) {
            case 0:

              // Test the constant distribution.
              StressTest();
              break;
            case 1:

              // Test the Gaussian distribution.
              StressTest();
              break;
          }
        }
      }
      return 0;
    }

    int StressTest() {

      typedef core::table::Table <
      core::tree::GenMetricTree <
      core::tree::AbstractStatistic > > TableType;

      // The list of arguments.
      std::vector< std::string > args;

      // Push in the reference dataset name.
      std::string attributes_in("random_attributes.csv");
      args.push_back(std::string("--attributes_in=") + attributes_in);

      // Push in the discrete choice set info name.
      std::string discrete_choice_set_info_in(
        "random_discrete_choice_set_info.csv");
      args.push_back(
        std::string("--discrete_choice_set_info_in=") +
        discrete_choice_set_info_in);

      // Print out the header of the trial.
      std::cout << "\n==================\n";
      std::cout << "Test trial begin\n";
      std::cout << "Number of attributes: " <<
                mlpack::mixed_logit_dcm::num_attributes_ << "\n";
      std::cout << "Number of people: " <<
                mlpack::mixed_logit_dcm::num_people_ << "\n";

      // Generate the random dataset and save it.
      TableType random_attribute_table;
      TableType random_discrete_choice_set_info_table;
      GenerateRandomDataset_(
        &random_attribute_table, &random_discrete_choice_set_info_table);
      random_attribute_table.Save(attributes_in);
      random_discrete_choice_set_info_table.Save(discrete_choice_set_info_in);

      return 0;
    };
};
}
}

BOOST_AUTO_TEST_SUITE(TestSuiteMixedLogitDCM)
BOOST_AUTO_TEST_CASE(TestCaseMixedLogitDCM) {

  // Call the tests.
  mlpack::mixed_logit_dcm::TestMixedLogitDCM dcm_test;
  dcm_test.StressTestMain();

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
