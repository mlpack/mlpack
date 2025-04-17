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
  std::vector<std::string> testFiles;
  testFiles.push_back("fake.csv");
  testFiles.push_back("german.csv");
  testFiles.push_back("iris.csv");
  testFiles.push_back("vc2.csv");
  testFiles.push_back("johnson8-4-4.csv");
  testFiles.push_back("lars_dependent_y.csv");
  testFiles.push_back("vc2_test_labels.txt");

  for (size_t i = 0; i < testFiles.size(); ++i)
  {
    arma::mat one, two;
    data::CSVOptions opts;
    opts.Fatal() = false;
    opts.NoTranspose() = false; // Transpose = true;
    opts.Categorical() = true;
    opts.FileFormat() = FileType::CSVASCII; 

    //data::Load(testFiles[i], one, data::NoFatal | data::Transpose | data::BIN_SER);
    data::Load(testFiles[i], one, data::NoFatal | data::Transpose | data::CSV);
    data::Load(testFiles[i], two, opts);

    // The following are passing correctly and throwing an error as it should
    // be
    // data::Load(testFiles[i], one, data::NoFatal | data::Transpose | data::MissingToNan | data::BIN_SER);
    // data::Load(testFiles[i], one, data::NoFatal | data::Transpose | data::Categorical | data::BIN_SER);
    // data::Load(testFiles[i], one, data::NoFatal | data::Transpose |  data::BIN_SER | data::Categorical);
    //data::Load(testFiles[i], one, data::BIN_SER | data::Categorical | data::NoFatal | data::Transpose);

    // Check that the matrices contain the same information.
    std::cout << one.n_elem << " == " << two.n_elem << std::endl;
    std::cout << one.n_rows << " == " << two.n_rows << std::endl;
    std::cout << one.n_cols << " == " << two.n_cols << std::endl;
  }
}

int main()
{
  //std::fstream f;
  //f.open("test.csv", std::fstream::out);
  //f << "1, 2, 3, 4" << std::endl;
  //f << "5, 6, 7" << std::endl;
  //f << "8, 9, 10, 11" << std::endl;
  //f.close();
  //// Here as well ? missing to nan
  //arma::mat dataset;
  //data::CSVOptions opts;
  //opts.Fatal() = false;
  //opts.NoTranspose() = true;
  //opts.Categorical() = true;

  //bool result = data::Load("test.csv", dataset, opts);

  //std::cout << "result: " << result << std::endl;
  //
  //
  test();

  //data::DataOptions opts;
  //opts.Fatal() = true;
  //opts.NoTranspose() = false;
  //opts.Categorical() = true;

  //arma::Mat<unsigned char> test(10, 15, arma::fill::randu);
  
  //inplace_trans(test);

  //test.brief_print();

  //data::Save("test.csv", test, opts);
  //arma::Mat<unsigned char> image, images;
  //arma::mat datamat;
  //ImageInfo info;
  
  //bool var = data::Load("vc2.csv", datamat, opts);
  ////bool var = data::Load("braziltourism.arff", datamat, opts);

  //datamat.brief_print();
  ////arma::Row<double> datarow;
  ////bool var = data::Load("vc2_labels.txt", datarow, opts);

  ////std::cout << " after loading.." << var << std::endl;
  //datarow.brief_print();


}
