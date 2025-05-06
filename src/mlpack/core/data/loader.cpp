#include <mlpack.hpp>
#include <chrono>
#include "load.hpp"

using namespace mlpack;
using namespace mlpack::data;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

void test()
{
  std::fstream f;
  f.open("test.arff", std::fstream::out);
  f << "@relation test" << std::endl;
  f << std::endl;
  f << "@attribute one STRING" << std::endl;
  f << "@attribute two REAL" << std::endl;
  f << std::endl;
  f << "@attribute three STRING" << std::endl;
  f << std::endl;
  f << "% a comment line " << std::endl;
  f << std::endl;
  f << "@data" << std::endl;
  f << "hello, 1, moo" << std::endl;
  f << "cheese, 2.34, goodbye" << std::endl;
  f << "seven, 1.03e+5, moo" << std::endl;
  f << "hello, -1.3, goodbye" << std::endl;
  f.close();

  arma::mat dataset;
  DatasetInfo info;
  if (!data::Load("test.arff", dataset, info))
    std::cout << "Cannot load dataset" << std::endl;

  std::cout << info.Dimensionality() << "==" << 3 << std::endl;
  std::cout << info.Type(0) << "==" << Datatype::categorical << std::endl;
  std::cout << info.NumMappings(0) << "==" << 3 << std::endl;
  std::cout << info.Type(1) << "==" << Datatype::numeric << std::endl;
  std::cout << info.Type(2) << "==" << Datatype::categorical << std::endl;
  std::cout << info.NumMappings(2) << "==" << 2 << std::endl;

  std::cout << dataset.n_rows << "==" << 3 << std::endl;
  std::cout << dataset.n_cols << "==" << 4 << std::endl;

  // The first dimension must all be different (except the ones that are the
  // same).
  std::cout << dataset(0, 0) << "==" << dataset(0, 3) << std::endl;
  std::cout << dataset(2, 0) << "==" << dataset(2, 2) << std::endl;
  std::cout << dataset(2, 1) << "==" << dataset(2, 3) << std::endl;

  remove("test.arff");
}


void test_header()
{
  std::fstream f;
  f.open("test.csv", std::fstream::out);
  f << "a, b, c, d" << std::endl;
  f << "1, 2, 3, 4" << std::endl;
  f << "5, 6, 7, 8" << std::endl;

  arma::mat dataset;
  data::Load("test.csv", dataset);

  std::cout << dataset.n_rows << "==" << 4 << std::endl;
  std::cout << dataset.n_cols << "==" << 2 << std::endl;
}

void test_imputer()
{
  std::fstream f;
  f.open("test_file.csv", std::fstream::out);
  f << "a, 2, 3"  << std::endl;
  f << "5, 6, a "  << std::endl;
  f << "8, 9, 10" << std::endl;
  f.close();

  arma::mat input;
  MissingPolicy policy({"a"});
  DatasetMapper<MissingPolicy> info(policy);
  data::Load("test_file.csv", input, info);

  // row and column test.
  std::cout << input.n_rows << "==" << "3" << std::endl;
  std::cout << input.n_cols << "==" << "3" << std::endl;

  // Load check
  // MissingPolicy should convert strings to nans.
  std::cout << std::isnan(input(0, 0)) << "==" << "true" << std::endl;
  std::cout << std::isnan(input(2, 1)) << "==" << "true" << std::endl;

  input.print();
  // convert missing vals to 99.
  CustomImputation<double> customStrategy(99);
  Imputer<double,
          DatasetMapper<MissingPolicy>,
          CustomImputation<double>> imputer(info, customStrategy);
  // convert a or nan to 99 for dimension 0.
  imputer.Impute(input, "a", 0);

  // Custom imputation result check.
  std::cout << std::isnan(input(2, 1)) << "==" << "true" << std::endl; // remains as NaN

  // Remove the file.
  remove("test_file.csv");
}

int main()
{
  //test_header();
  test_imputer();
}
