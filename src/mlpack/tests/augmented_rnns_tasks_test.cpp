/**
 * @file augmented_rnns_tasks.cpp
 * @author Konstantin Sidorov
 *
 * Tests the rtasks for augmented network models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <iostream>

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/augmented/tasks/copy.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using std::cerr;

using namespace mlpack::ann::augmented::tasks;

// The dummy model that simply copies the sequence the required number of times
// (yes, no ML here, we're unit testing :)
class HardCodedCopyModel {
public:
  HardCodedCopyModel() : nRepeats(1) {}
  void Train(
      arma::field<arma::irowvec>& predictors,
      arma::field<arma::irowvec>& labels) {
    auto input = predictors.at(0);
    auto output = labels.at(0);
    nRepeats = output.n_elem / input.n_elem;
  }
  void Predict(
      arma::irowvec& predictors,
      arma::irowvec& labels) {
    int len = predictors.n_elem;
    auto totalLen = nRepeats*len;
    labels.zeros(totalLen);
    for (int i = 0; i < totalLen; ++i) {
      labels.at(i) = predictors.at(i % len);
    }
  }
private:
  int nRepeats;
};

BOOST_AUTO_TEST_SUITE(AugmentedRNNsTasks);

BOOST_AUTO_TEST_CASE(CopyTaskTest)
{
  bool ok = true;
  // Check the setup on vrious lengths...
  for (int maxLen = 2; maxLen <= 16; ++maxLen) {
    // .. and various numbers of repetitions.
    for (int nRepeats = 1; nRepeats <= 10; ++nRepeats) {
      CopyTask task(maxLen, nRepeats);
      HardCodedCopyModel model;
      // A single failure is a failure.
      if (task.Evaluate(model) < 0.99) {
        ok = false;
        break;
      }
    }
    if (!ok) break;
  }
  BOOST_REQUIRE(ok);
}

BOOST_AUTO_TEST_SUITE_END();