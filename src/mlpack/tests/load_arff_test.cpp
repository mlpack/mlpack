/*
  @file load_arff_test.cpp
  @author Praveen Ch

  A test to check whether the arff loader is case insensitive to declarations:
  @relation, @attribute, @data.

*/

#include <mlpack/core/data/load_arff.hpp>
#include <mlpack/core/data/dataset_mapper.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/data/map_policies/increment_policy.hpp>

#include <boost/algorithm/string/trim.hpp>
#include <unordered_map>
#include <boost/bimap.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack::data;

BOOST_AUTO_TEST_SUITE(ARFFLoaderTest);

BOOST_AUTO_TEST_CASE(CaseTest)
{
  arma::mat dataset;

  DatasetMapper<IncrementPolicy> info;

  LoadARFF<double, IncrementPolicy>("casecheck.arff", dataset, info);

  BOOST_CHECK_EQUAL(dataset.n_rows, 2);
  BOOST_CHECK_EQUAL(dataset.n_cols, 3);

}

BOOST_AUTO_TEST_SUITE_END();