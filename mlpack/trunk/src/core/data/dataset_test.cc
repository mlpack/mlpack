#include <armadillo>

#include "dataset.h"


using arma::mat;

#define BOOST_TEST_MODULE DataSetTest 
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TestSplitTrainTest) {

  Dataset orig;
  
  orig.InitBlank();
  orig.matrix().set_size(1, 12);
  orig.info().InitContinuous(1);
  
  for (int i = 0; i < 12; i++) {
    orig.matrix()(0, i) = i;
  }
  
  std::vector<int> found;
  for (int i = 0; i < 12; i++)
    found.push_back(0);
  
  Dataset train;
  Dataset test;
  
  // TODO: replace use of std::vector with arma::vec (fixed-size container)
  std::vector<size_t> permutation;
  for(int i = 0; i < 12; i++)
    permutation.push_back(i);
  
  orig.SplitTrainTest(4, 1, permutation,
      train, test);
   
   BOOST_REQUIRE(test.n_points() == 3); 
   BOOST_REQUIRE(train.n_points() == 9);

   BOOST_REQUIRE_CLOSE(test.get(0, 0), 1.0, 1e-5);
   BOOST_REQUIRE_CLOSE(test.get(0, 1), 5.0, 1e-5);
   BOOST_REQUIRE_CLOSE(test.get(0, 2), 9.0, 1e-5);

   BOOST_REQUIRE_CLOSE(train.get(0, 3), 4.0, 1e-5);
   BOOST_REQUIRE_CLOSE(train.get(0, 4), 6.0, 1e-5);
   BOOST_REQUIRE_CLOSE(train.get(0, 5), 7.0, 1e-5);
   BOOST_REQUIRE_CLOSE(train.get(0, 6), 8.0, 1e-5);
   BOOST_REQUIRE_CLOSE(train.get(0, 7), 10.0, 1e-5);
   BOOST_REQUIRE_CLOSE(train.get(0, 8), 11.0, 1e-5);

}

void AssertSameMatrix(const mat& a, const mat& b) {
  size_t r = a.n_rows;
  size_t c = a.n_cols; 
   
  BOOST_REQUIRE_CLOSE(a.n_rows + 1e-6, b.n_rows + 1e-6, 1e-5); 
  BOOST_REQUIRE_CLOSE(a.n_cols + 1e-6, b.n_cols + 1e-6, 1e-5);

  for (size_t ri = 0; ri < r; ri++) {
    for (size_t ci = 0; ci < c; ci++) {
       BOOST_REQUIRE_CLOSE(a(ri,ci) + 1e-6, b(ri,ci) + 1e-6, 1e-5);
    }
  }
}

BOOST_AUTO_TEST_CASE(TestLoad) {

  Dataset d1;
  Dataset d2;
  Dataset d3;
  Dataset d4;
  Dataset d5;
 
  BOOST_REQUIRE(d1.InitFromFile("fake.arff") == true);
  BOOST_REQUIRE(d2.InitFromFile("fake.csv") == true);
  BOOST_REQUIRE(d3.InitFromFile("fake.csvh") == true);
  BOOST_REQUIRE(d4.InitFromFile("fake.tsv") == true);
  BOOST_REQUIRE(d5.InitFromFile("fake.weird") == true);
 
  AssertSameMatrix(d1.matrix(), d2.matrix());
  AssertSameMatrix(d1.matrix(), d3.matrix());
  AssertSameMatrix(d1.matrix(), d4.matrix());
  AssertSameMatrix(d1.matrix(), d5.matrix());
}


BOOST_AUTO_TEST_CASE(TestStoreLoad) {
  Dataset d1;
  Dataset d2;
  Dataset d3;
 
  BOOST_REQUIRE(d1.InitFromFile("fake.arff") == true);
  
  d1.WriteCsv("fakeout1.csv");
  d1.WriteArff("fakeout1.arff");

  BOOST_REQUIRE(d2.InitFromFile("fakeout1.arff") == true);
  BOOST_REQUIRE(d3.InitFromFile("fakeout1.csv") == true);

  AssertSameMatrix(d1.matrix(), d2.matrix());
  AssertSameMatrix(d1.matrix(), d3.matrix());
  
  BOOST_REQUIRE_CLOSE(strcmp(d1.info().name().c_str(), d2.info().name().c_str()) 
	+ 1e-6, 1e-6, 1e-5);

  for (size_t i = 0; i < d1.info().n_features(); i++) {
    BOOST_REQUIRE_EQUAL(d1.info().feature(i).name(), d2.info().feature(i).name());
    BOOST_REQUIRE_EQUAL(d1.info().feature(i).type(), d2.info().feature(i).type());
  }
}


/*
int main(int argc, char *argv[]) {
  xrun_init(argc, argv);
  const char *in = xrun_param_str("in");
  const char *out = xrun_param_str("out");
  String type;
  
  type.Copy(xrun_param_str("type"));
  
  Dataset dataset;
  
  if (!(dataset.InitFromFile(in))) return 1;
  
  bool result;
  
  if (type.EqualsNoCase("arff")) {
    result = dataset.WriteArff(out);
  } else if (type.EqualsNoCase("csv")) {
    result = dataset.WriteCsv(out, false);
  } else if (type.EqualsNoCase("csvh")) {
    result = dataset.WriteCsv(out, true);
  } else {
    result = false;
  }
  
  if (!(result)) {
    fprintf(stderr, "Error!\n");
    return 1;
  }
  
  std::vector<size_t> permutation;
  math::MakeRandomPermutation(dataset.n_points(), &permutation);
  
  for (int k = 5; k < 10; k++) {
    Dataset test;
    Dataset train;
    int i = k - 5;
    dataset.SplitTrainTest(k, i, permutation, &train, &test);
    assert(test.n_points() + train.n_points() == dataset.n_points());
  }

  return 0;
}
*/
