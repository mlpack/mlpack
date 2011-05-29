/** @file monte_carlo.test.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <time.h>
#include "core/monte_carlo/mean_variance_pair.h"

BOOST_AUTO_TEST_SUITE(TestSuiteMonteCarlo)
BOOST_AUTO_TEST_CASE(TestCaseMonteCarlo) {

  core::monte_carlo::MeanVariancePair mv;
  int num_random_numbers = core::math::RandInt(50, 100);
  int subtract = core::math::RandInt(10, 20);
  int num_subset = num_random_numbers - subtract;
  std::vector<double> random_numbers(num_random_numbers, 0.0);
  core::monte_carlo::MeanVariancePair mv_subset;
  for(int i = 0; i < num_random_numbers; i++) {
    random_numbers[i] = core::math::Random(-5.0, 5.0);
    mv.push_back(random_numbers[i]);
    if(i < num_subset) {
      mv_subset.push_back(random_numbers[i]);
    }
  }

  // Now remove the last two numbers.
  for(int i = 0; i < subtract; i++) {
    mv.pop(random_numbers[ num_random_numbers - i - 1]);
  }

  std::cout << "Generating " << num_random_numbers << " numbers...\n";
  std::cout << "Accumulating statistics up to " << num_subset <<
            " numbers...\n";
  std::cout << "Decremental update result: mean: " << mv.sample_mean() <<
            ", variance: " << mv.sample_variance() << "\n";
  std::cout << "Compare: mean: " << mv_subset.sample_mean() << ", variance: " <<
            mv_subset.sample_variance() << "\n";

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
