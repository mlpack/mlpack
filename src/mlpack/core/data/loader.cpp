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
  //testFiles.push_back("lars_kkt.bin");
   // testFiles.push_back("german.csv");
   testFiles.push_back("german.csv");
  // testFiles.push_back("iris.csv");
  // testFiles.push_back("vc2.csv");
  // testFiles.push_back("johnson8-4-4.csv");
  // testFiles.push_back("lars_dependent_y.csv");
  // testFiles.push_back("vc2_test_labels.txt");

  for (size_t i = 0; i < testFiles.size(); ++i)
  {
    //arma::mat one, two;
    //data::CSVOptions opts;
    //opts.Fatal() = false;
    //opts.NoTranspose() = false; // Transpose = true;
    //opts.Categorical() = true;
    //opts.FileFormat() = FileType::CSVASCII; 

    //std::cout << "file type" << std::endl;
    //data::Load(testFiles[i], one, data::NoFatal | data::Transpose | data::BIN_SER);
    
    //arma::mat inputData;

    //if (!data::Load("vc2.csv", inputData))
      //std::cout << "Cannot load train dataset vc2.csv!" << std::endl;

    std::fstream f;
    f.open("test.csv", std::fstream::out);
    f << "1, 2, hello" << std::endl;
    f << "3, 4, goodbye" << std::endl;
    f << "5, 6, coffee" << std::endl;
    f << "7, 8, confusion" << std::endl;
    f << "9, 10, hello" << std::endl;
    f << "11, 12, confusion" << std::endl;
    f << "13, 14, confusion" << std::endl;
    f.close();

    // Load the test CSV.
    arma::umat matrix;
    data::CSVOptions opts;
    opts.Fatal() = false;
    opts.NoTranspose() = false; // Transpose = true;
    opts.Categorical() = true;

    if (!data::Load("test.csv", matrix, opts))
      std::cout <<"Cannot load dataset test.csv" << std::endl;

    arma::umat output;
    matrix.print();
    data::OneHotEncoding(matrix, output, opts.Mapper());
    std::cout << output.n_cols << " == " << 7 << std::endl;
    std::cout << output.n_rows << " == " << 6 << std::endl;
    output.brief_print();
    std::cout << opts.Mapper().Type(0) << std::endl;  //Datatype::numeric;
    std::cout << opts.Mapper().Type(1) << std::endl;  //Datatype::numeric;
    std::cout << opts.Mapper().Type(2) << std::endl;  //Datatype::categorical;




    //std::fstream f;
    //f.open("test_sparse_file.tsv", std::fstream::out);

    //f << "1\t2\t0.1" << std::endl;
    //f << "2\t3\t0.2" << std::endl;
    //f << "3\t4\t0.3" << std::endl;
    //f << "4\t5\t0.4" << std::endl;
    //f << "5\t6\t0.5" << std::endl;
    //f << "6\t7\t0.6" << std::endl;
    //f << "7\t8\t0.7" << std::endl;

    //f.close();

    //arma::sp_mat test;

    //std::cout<< "is arma:: " << IsSparseMat<arma::sp_mat>::value << std::endl;

    //data::Load("test_sparse_file.tsv", test,
        //data::Fatal | data::NoTranspose);

    ////test.n_rows == 8;
    ////test.n_cols == 9;

    //remove("test_sparse_file.tsv");

    //// Should the user be allowed to load a model into an armadillo matrix?
    //data::Load(testFiles[i], one, data::NoFatal | data::Transpose);
    //data::Load(testFiles[i], two, opts);

    // The following are passing correctly and throwing an error as it should
    // be
    // data::Load(testFiles[i], one, data::NoFatal | data::Transpose | data::MissingToNan | data::BIN_SER);
    // data::Load(testFiles[i], one, data::NoFatal | data::Transpose | data::Categorical | data::BIN_SER);
    // data::Load(testFiles[i], one, data::NoFatal | data::Transpose |  data::BIN_SER | data::Categorical);
    //data::Load(testFiles[i], one, data::BIN_SER | data::Categorical | data::NoFatal | data::Transpose);

    // Check that the matrices contain the same information.
    //std::cout << one.n_elem << " == " << two.n_elem << std::endl;
    //std::cout << one.n_rows << " == " << two.n_rows << std::endl;
    //std::cout << one.n_cols << " == " << two.n_cols << std::endl;
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
