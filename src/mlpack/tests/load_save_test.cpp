/**
 * @file tests/load_save_test.cpp
 * @author Ryan Curtin
 *
 * Tests for Load() and Save().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <sstream>

#include <mlpack/core.hpp>
#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace std;

#define MLPACK_ENABLE_HTTPLIB

/**
 * Make sure failure occurs when no extension given.
 */
TEST_CASE("NoExtensionLoad", "[LoadSaveTest][tiny]")
{
  arma::mat out;
  REQUIRE(Load("noextension", out) == false);
}

/**
 * Make sure failure occurs when no extension given.
 */
TEST_CASE("NoExtensionSave", "[LoadSaveTest][tiny]")
{
  arma::mat out;
  REQUIRE(Save("noextension", out) == false);
}

/**
 * Make sure load fails if the file does not exist.
 */
TEST_CASE("NotExistLoad", "[LoadSaveTest][tiny]")
{
  arma::mat out;
  REQUIRE(Load("nonexistentfile_______________.csv", out) == false);
}

/**
 * Make sure load fails if the file extension is wrong in automatic detection mode.
 */
TEST_CASE("WrongExtensionWrongLoad", "[LoadSaveTest][tiny]")
{
  // Try to load arma::arma_binary file with ".csv" extension
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.save("test_file.csv", arma::arma_binary) == true);

  // Now reload through our interface.
  REQUIRE(Load("test_file.csv", test) == false);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure load is successful even if the file extension is wrong when file type is specified.
 */
TEST_CASE("WrongExtensionCorrectLoad", "[LoadSaveTest][tiny]")
{
  // Try to load arma::arma_binary file with ".csv" extension
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.save("test_file.csv", arma::arma_binary) == true);

  // Now reload through our interface.
  REQUIRE(Load("test_file.csv", test, ArmaBin)
      == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; i++)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-3));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure a CSV is loaded correctly.
 */
TEST_CASE("LoadCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.csv", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure a TSV is loaded correctly to a sparse matrix.
 */
TEST_CASE("LoadSparseTSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_sparse_file.tsv", fstream::out);

  f << "1\t2\t0.1" << endl;
  f << "2\t3\t0.2" << endl;
  f << "3\t4\t0.3" << endl;
  f << "4\t5\t0.4" << endl;
  f << "5\t6\t0.5" << endl;
  f << "6\t7\t0.6" << endl;
  f << "7\t8\t0.7" << endl;

  f.close();

  arma::sp_mat test;

  REQUIRE(Load(
      "test_sparse_file.tsv", test, Fatal + NoTranspose) == true);

  REQUIRE(test.n_rows == 8);
  REQUIRE(test.n_cols == 9);

  arma::sp_mat::const_iterator it = test.begin();
  arma::sp_mat::const_iterator it_end = test.end();

  double temp = 0.1;
  for (int i = 0; it != it_end; ++it, temp += 0.1, ++i)
  {
    REQUIRE((double)(*it) == Approx(temp).epsilon(1e-7));
    REQUIRE((int)(it.row()) == i + 1);
    REQUIRE((int)it.col() == i + 2);
  }
  // Remove the file.
  remove("test_sparse_file.tsv");
}

/**
 * Make sure a CSV in text format is loaded correctly to a sparse matrix.
 */
TEST_CASE("LoadSparseTXTTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_sparse_file.txt", fstream::out);

  f << "1 2 0.1" << endl;
  f << "2 3 0.2" << endl;
  f << "3 4 0.3" << endl;
  f << "4 5 0.4" << endl;
  f << "5 6 0.5" << endl;
  f << "6 7 0.6" << endl;
  f << "7 8 0.7" << endl;

  f.close();

  arma::sp_mat test;

  REQUIRE(Load("test_sparse_file.txt", test, Fatal + NoTranspose)
      == true);

  REQUIRE(test.n_rows == 8);
  REQUIRE(test.n_cols == 9);

  arma::sp_mat::const_iterator it = test.begin();
  arma::sp_mat::const_iterator it_end = test.end();

  double temp = 0.1;
  for (int i = 0; it != it_end; ++it, temp += 0.1, ++i)
  {
    REQUIRE((double) (*it) == Approx(temp).epsilon(1e-7));
    REQUIRE((int) (it.row()) == i + 1);
    REQUIRE((int) it.col() == i + 2);
  }
  // Remove the file.
  remove("test_sparse_file.txt");
}

/**
 * Make sure sparse coordinate list autodetection works.
 */
TEST_CASE("LoadSparseAutodetectTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 3, 4.0" << endl;
  f << "2, 4, 5.0" << endl;
  f << "3, 6, -3.0" << endl;

  f.close();

  arma::sp_mat test;

  REQUIRE(Load("test_file.csv", test, Fatal + Transpose)
      == true);

  REQUIRE(test.n_rows == 7);
  REQUIRE(test.n_cols == 4);

  REQUIRE(test.at(3, 1) == Approx(4.0));
  REQUIRE(test.at(4, 2) == Approx(5.0));
  REQUIRE(test.at(6, 3) == Approx(-3.0));

  remove("test_file.csv");
}

/**
 * Make sure sparse coordinate list autodetection fails when the number of
 * columns is wrong.
 */
TEST_CASE("LoadSparseAutodetectNotCoordinateListTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 0, 0, 4" << endl;
  f << "0, 1, 0, 3" << endl;
  f << "1, 0, 0, 0" << endl;

  f.close();

  arma::sp_mat test;

  REQUIRE(Load("test_file.csv", test, Fatal + Transpose) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 3);

  REQUIRE(test.at(0, 0) == Approx(1.0));
  REQUIRE(test.at(3, 0) == Approx(4.0));
  REQUIRE(test.at(1, 1) == Approx(1.0));
  REQUIRE(test.at(3, 1) == Approx(3.0));
  REQUIRE(test.at(0, 2) == Approx(1.0));

  remove("test_file.csv");
}

/**
 * Make sure a TSV is loaded correctly.
 */
TEST_CASE("LoadTSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.csv", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Test TSV loading with .tsv extension.
 */
TEST_CASE("LoadTSVExtensionTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.tsv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.tsv", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.tsv");
}

/**
 * Test that we can manually specify the format for loading.
 */
TEST_CASE("LoadAnyExtensionFileTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.blah", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.blah", test, RawAscii));

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.blah");
}

/**
 * Make sure a CSV is saved correctly.
 */
TEST_CASE("SaveCSVTest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(Save("test_file.csv", test) == true);

  // Load it in and make sure it is the same.
  arma::mat test2;
  REQUIRE(Load("test_file.csv", test2) == true);

  REQUIRE(test2.n_rows == 4);
  REQUIRE(test2.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test2[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure a TXT is saved correctly for a sparse matrix
 */
TEST_CASE("SaveSparseTXTTest", "[LoadSaveTest][tiny]")
{
  arma::sp_mat test = "0.1 0 0 0;"
                      "0 0.2 0 0;"
                      "0 0 0.3 0;"
                      "0 0 0 0.4;";

  REQUIRE(Save("test_sparse_file.txt", test,
        Fatal + Transpose) == true);

  // Load it in and make sure it is the same.
  arma::sp_mat test2;
  REQUIRE(Load("test_sparse_file.txt", test2,
        Fatal + Transpose) == true);

  REQUIRE(test2.n_rows == 4);
  REQUIRE(test2.n_cols == 4);

  arma::sp_mat::const_iterator it = test2.begin();
  arma::sp_mat::const_iterator it_end = test2.end();

  double temp = 0.1;
  for (int i = 0; it != it_end; ++it, temp += 0.1, ++i)
  {
    double val = (*it);
    REQUIRE(val == Approx(temp).epsilon(1e-7));
    REQUIRE((int)(it.row()) == i);
    REQUIRE((int)it.col() == i);
  }

  // Remove the file.
  remove("test_sparse_file.txt");
}

/**
 * Make sure a Sparse Matrix is saved and loaded correctly in binary format
 */
TEST_CASE("SaveSparseBinaryTest", "[LoadSaveTest][tiny]")
{
  arma::sp_mat test = "0.1 0 0 0;"
                      "0 0.2 0 0;"
                      "0 0 0.3 0;"
                      "0 0 0 0.4;";

  REQUIRE(Save("test_sparse_file.bin", test, Fatal + NoTranspose)
      == true);

  // Load it in and make sure it is the same.
  arma::sp_mat test2;
  REQUIRE(Load("test_sparse_file.bin", test2, Fatal + NoTranspose)
      == true);

  REQUIRE(test2.n_rows == 4);
  REQUIRE(test2.n_cols == 4);

  arma::sp_mat::const_iterator it = test2.begin();
  arma::sp_mat::const_iterator it_end = test2.end();

  double temp = 0.1;
  for (int i = 0; it != it_end; ++it, temp += 0.1, ++i)
  {
    double val = (*it);
    REQUIRE(val == Approx(temp).epsilon(1e-7));
    REQUIRE((int) (it.row()) == i);
    REQUIRE((int) it.col() == i);
  }

  // Remove the file.
  remove("test_sparse_file.bin");
}

/**
 * Make sure CSVs can be loaded in transposed form.
 */
TEST_CASE("LoadTransposedCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.csv", test) == true);

  REQUIRE(test.n_cols == 2);
  REQUIRE(test.n_rows == 4);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure ColVec can be loaded.
 */
TEST_CASE("LoadColVecCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  for (size_t i = 0; i < 8; ++i)
    f << i << endl;

  f.close();

  arma::colvec test;
  REQUIRE(Load("test_file.csv", test) == true);

  REQUIRE(test.n_cols == 1);
  REQUIRE(test.n_rows == 8);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) i).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure we can load a transposed column vector.
 */
TEST_CASE("LoadColVecTransposedCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  for (size_t i = 0; i < 8; ++i)
    f << i << ", ";
  f << "8" << endl;
  f.close();

  arma::colvec test;
  REQUIRE(Load("test_file.csv", test) == true);

  REQUIRE(test.n_cols == 1);
  REQUIRE(test.n_rows == 9);

  for (size_t i = 0; i < 9; ++i)
    REQUIRE(test[i] == Approx((double) i).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure besides numeric data "quoted strings" or
 * 'quoted strings' in csv files are loaded correctly.
 */
TEST_CASE("LoadQuotedStringInCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1,field 2,field 3" << endl;
  f << "2,\"field 2, with comma\",field 3" << endl;
  f << "3,field 2 with \"embedded quote\",field 3" << endl;
  f << "4, field 2 with embedded \\ ,field 3" << endl;
  f << "5, ,field 3" << endl;

  f.close();

  std::vector<std::string> elements;
  elements.push_back("field 2");
  elements.push_back("\"field 2, with comma\"");
  elements.push_back("field 2 with \"embedded quote\"");
  elements.push_back("field 2 with embedded \\");
  elements.push_back("");

  arma::mat test;
  TextOptions opts = Categorical;
  REQUIRE(Load("test_file.csv", test, opts) == true);

  REQUIRE(test.n_rows == 3);
  REQUIRE(test.n_cols == 5);
  REQUIRE(opts.DatasetInfo().Dimensionality() == 3);

  // Check each element for equality/ closeness.
  for (size_t i = 0; i < 5; ++i)
    REQUIRE(test.at(0, i) == Approx((double) (i + 1)).epsilon(1e-7));

  for (size_t i = 0; i < 5; ++i)
    REQUIRE(opts.DatasetInfo().UnmapString(test.at(1, i), 1, 0)
        == elements[i]);

  for (size_t i = 0; i < 5; ++i)
    REQUIRE(opts.DatasetInfo().UnmapString(test.at(2, i), 2, 0) == "field 3");

  // Clear the vector to free the space.
  elements.clear();
  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure besides numeric data "quoted strings" or
 * 'quoted strings' in txt files are loaded correctly.
 */
TEST_CASE("LoadQuotedStringInTXTTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.txt", fstream::out);

  f << "1 field2 field3" << endl;
  f << "2 \"field 2 with space\" field3" << endl;

  f.close();

  std::vector<std::string> elements;
  elements.push_back("field2");
  elements.push_back("\"field 2 with space\"");

  arma::mat test;
  TextOptions opts = Categorical;
  REQUIRE(Load("test_file.txt", test, opts) == true);

  REQUIRE(test.n_rows == 3);
  REQUIRE(test.n_cols == 2);
  REQUIRE(opts.DatasetInfo().Dimensionality() == 3);

  // Check each element for equality/ closeness.
  for (size_t i = 0; i < 2; ++i)
    REQUIRE(test.at(0, i) == Approx((double) (i + 1)).epsilon(1e-7));

  for (size_t i = 0; i < 2; ++i)
    REQUIRE(opts.DatasetInfo().UnmapString(test.at(1, i), 1, 0)
        == elements[i]);

  for (size_t i = 0; i < 2; ++i)
    REQUIRE(opts.DatasetInfo().UnmapString(test.at(2, i), 2, 0) == "field3");

  // Clear the vector to free the space.
  elements.clear();
  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure besides numeric data "quoted strings" or
 * 'quoted strings' in tsv files are loaded correctly.
 */
TEST_CASE("LoadQuotedStringInTSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.tsv", fstream::out);

  f << "1\tfield 2\tfield 3" << endl;
  f << "2\t\"field 2\t with tab\"\tfield 3" << endl;
  f << "3\tfield 2 with \"embedded quote\"\tfield 3" << endl;
  f << "4\t field 2 with embedded \\ \tfield 3" << endl;
  f << "5\t \tfield 3" << endl;

  f.close();

  std::vector<std::string> elements;
  elements.push_back("field 2");
  elements.push_back("\"field 2\t with tab\"");
  elements.push_back("field 2 with \"embedded quote\"");
  elements.push_back("field 2 with embedded \\");
  elements.push_back("");

  arma::mat test;
  TextOptions opts = Categorical;

  REQUIRE(Load("test_file.tsv", test, opts) == true);

  REQUIRE(test.n_rows == 3);
  REQUIRE(test.n_cols == 5);
  REQUIRE(opts.DatasetInfo().Dimensionality() == 3);

  // Check each element for equality/ closeness.
  for (size_t i = 0; i < 5; ++i)
    REQUIRE(test.at(0, i) == Approx((double) (i + 1)).epsilon(1e-7));

  for (size_t i = 0; i < 5; ++i)
    REQUIRE(opts.DatasetInfo().UnmapString(test.at(1, i), 1, 0)
        == elements[i]);

  for (size_t i = 0; i < 5; ++i)
    REQUIRE(opts.DatasetInfo().UnmapString(test.at(2, i), 2, 0) == "field 3");

  // Clear the vector to free the space.
  elements.clear();
  // Remove the file.
  remove("test_file.tsv");
}

/**
 * Make sure Load() throws an exception when trying to load a matrix into a
 * colvec or rowvec.
 */
TEST_CASE("LoadMatinVec", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 2" << endl;
  f << "3, 4" << endl;

  f.close();

  /**
   * Log::Fatal will be called when the matrix is not of the right size.
   */
  arma::vec coltest;
  REQUIRE_THROWS_AS(Load("test_file.csv", coltest, Fatal + Transpose),
      std::runtime_error);

  arma::rowvec rowtest;
  REQUIRE_THROWS_AS(Load("test_file.csv", rowtest, Fatal + Transpose),
      std::runtime_error);

  remove("test_file.csv");
}

/**
 * Make sure that rowvecs can be loaded successfully.
 */
TEST_CASE("LoadRowVecCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  for (size_t i = 0; i < 7; ++i)
    f << i << ", ";
  f << "7";
  f << endl;

  f.close();

  arma::rowvec test;
  REQUIRE(Load("test_file.csv", test) == true);

  REQUIRE(test.n_cols == 8);
  REQUIRE(test.n_rows == 1);

  for (size_t i = 0; i < 8 ; ++i)
    REQUIRE(test[i] == Approx((double) i).epsilon(1e-7));

  remove("test_file.csv");
}

/**
 * Make sure that we can load transposed row vectors.
 */
TEST_CASE("LoadRowVecTransposedCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  for (size_t i = 0; i < 8; ++i)
    f << i << endl;

  f.close();

  arma::rowvec test;
  REQUIRE(Load("test_file.csv", test) == true);

  REQUIRE(test.n_rows == 1);
  REQUIRE(test.n_cols == 8);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) i).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure TSVs can be loaded in transposed form.
 */
TEST_CASE("LoadTransposedTSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.csv", test) == true);

  REQUIRE(test.n_cols == 2);
  REQUIRE(test.n_rows == 4);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Check TSV loading with .tsv extension.
 */
TEST_CASE("LoadTransposedTSVExtensionTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.tsv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.tsv", test) == true);

  REQUIRE(test.n_cols == 2);
  REQUIRE(test.n_rows == 4);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.tsv");
}

/**
 * Make sure CSVs can be loaded in non-transposed form.
 */
TEST_CASE("LoadNonTransposedCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 3, 5, 7" << endl;
  f << "2, 4, 6, 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.csv", test, NoFatal + NoTranspose) == true);

  REQUIRE(test.n_cols == 4);
  REQUIRE(test.n_rows == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure CSVs can be saved in non-transposed form.
 */
TEST_CASE("SaveNonTransposedCSVTest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 2;"
                   "3 4;"
                   "5 6;"
                   "7 8;";

  REQUIRE(Save("test_file.csv", test, NoFatal + NoTranspose) == true);

  // Load it in and make sure it is in the same.
  arma::mat test2;
  REQUIRE(Load("test_file.csv", test2, NoFatal + NoTranspose) == true);

  REQUIRE(test2.n_rows == 4);
  REQUIRE(test2.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx(test2[i]).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure arma_ascii is loaded correctly.
 */
TEST_CASE("LoadArmaASCIITest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.save("test_file.txt", arma::arma_ascii));

  REQUIRE(Load("test_file.txt", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure a CSV is saved correctly.
 */
TEST_CASE("SaveArmaASCIITest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(Save("test_file.txt", test) == true);

  // Load it in and make sure it is the same.
  REQUIRE(Load("test_file.txt", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure raw_ascii is loaded correctly.
 */
TEST_CASE("LoadRawASCIITest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.txt", fstream::out);

  f << "1 2 3 4" << endl;
  f << "5 6 7 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.txt", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure CSV is loaded correctly as .txt.
 */
TEST_CASE("LoadCSVTxtTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test_file.txt", fstream::out);

  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(Load("test_file.txt", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure arma_binary is loaded correctly.
 */
TEST_CASE("LoadArmaBinaryTest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.save("test_file.bin", arma::arma_binary)
      == true);

  // Now reload through our interface.
  REQUIRE(Load("test_file.bin", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.bin");
}

/**
 * Make sure arma_binary is saved correctly.
 */
TEST_CASE("SaveArmaBinaryTest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(Save("test_file.bin", test) == true);

  REQUIRE(Load("test_file.bin", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.bin");
}

/**
 * Make sure that we can manually specify the format.
 */
TEST_CASE("SaveArmaBinaryArbitraryExtensionTest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(Save("test_file.blerp.blah", test, ArmaBin) == true);

  REQUIRE(Load("test_file.blerp.blah", test, ArmaBin) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.blerp.blah");
}

/**
 * Make sure raw_binary is loaded correctly.
 */
TEST_CASE("LoadRawBinaryTest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 2;"
                   "3 4;"
                   "5 6;"
                   "7 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.save("test_file.bin", arma::raw_binary)
      == true);

  // Now reload through our interface.
  REQUIRE(Load("test_file.bin", test) == true);

  REQUIRE(test.n_rows == 1);
  REQUIRE(test.n_cols == 8);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.bin");
}

/**
 * Make sure load as PGM is successful.
 */
TEST_CASE("LoadPGMBinaryTest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.save("test_file.pgm", arma::pgm_binary)
      == true);

  // Now reload through our interface.
  REQUIRE(Load("test_file.pgm", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.pgm");
}

/**
 * Make sure save as PGM is successful.
 */
TEST_CASE("SavePGMBinaryTest", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(Save("test_file.pgm", test) == true);

  // Now reload through our interface.
  REQUIRE(Load("test_file.pgm", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.pgm");
}

#if defined(ARMA_USE_HDF5)
/**
 * Make sure load as HDF5 is successful.
 */
TEST_CASE("LoadHDF5Test", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";
  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.save("test_file.h5", arma::hdf5_binary)
      == true);
  REQUIRE(testTrans.save("test_file.hdf5", arma::hdf5_binary)
      == true);
  REQUIRE(testTrans.save("test_file.hdf", arma::hdf5_binary)
      == true);
  REQUIRE(testTrans.save("test_file.he5", arma::hdf5_binary)
      == true);

  // Now reload through our interface.
  REQUIRE(Load("test_file.h5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Make sure the other extensions work too.
  REQUIRE(Load("test_file.hdf5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  REQUIRE(Load("test_file.hdf", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  REQUIRE(Load("test_file.he5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  remove("test_file.h5");
  remove("test_file.hdf");
  remove("test_file.hdf5");
  remove("test_file.he5");
}

/**
 * Make sure save as HDF5 is successful.
 */
TEST_CASE("SaveHDF5Test", "[LoadSaveTest][tiny]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";
  REQUIRE(Save("test_file.h5", test) == true);
  REQUIRE(Save("test_file.hdf5", test) == true);
  REQUIRE(Save("test_file.hdf", test) == true);
  REQUIRE(Save("test_file.he5", test) == true);

  // Now load them all and verify they were saved okay.
  REQUIRE(Load("test_file.h5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Make sure the other extensions work too.
  REQUIRE(Load("test_file.hdf5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  REQUIRE(Load("test_file.hdf", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  REQUIRE(Load("test_file.he5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  remove("test_file.h5");
  remove("test_file.hdf");
  remove("test_file.hdf5");
  remove("test_file.he5");
}

#endif

/**
 * Test normalization of labels.
 */
TEST_CASE("NormalizeLabelSmallDatasetTest", "[LoadSaveTest][tiny]")
{
  arma::irowvec labels("-1 1 1 -1 -1 -1 1 1");
  arma::Row<size_t> newLabels;
  arma::ivec mappings;

  NormalizeLabels(labels, newLabels, mappings);

  REQUIRE(mappings[0] == -1);
  REQUIRE(mappings[1] == 1);

  REQUIRE(newLabels[0] == 0);
  REQUIRE(newLabels[1] == 1);
  REQUIRE(newLabels[2] == 1);
  REQUIRE(newLabels[3] == 0);
  REQUIRE(newLabels[4] == 0);
  REQUIRE(newLabels[5] == 0);
  REQUIRE(newLabels[6] == 1);
  REQUIRE(newLabels[7] == 1);

  arma::irowvec revertedLabels;

  RevertLabels(newLabels, mappings, revertedLabels);

  for (size_t i = 0; i < labels.n_elem; ++i)
    REQUIRE(labels[i] == revertedLabels[i]);
}

/**
 * Harder label normalization test.
 */
TEST_CASE("NormalizeLabelTest", "[LoadSaveTest][tiny]")
{
  arma::rowvec randLabels(5000);
  for (size_t i = 0; i < 5000; ++i)
    randLabels[i] = RandInt(-50, 50);
  randLabels[0] = 0.65; // Hey, doubles work too!

  arma::Row<size_t> newLabels;
  arma::vec mappings;

  NormalizeLabels(randLabels, newLabels, mappings);

  // Now map them back and ensure they are right.
  arma::rowvec revertedLabels(5000);
  RevertLabels(newLabels, mappings, revertedLabels);

  for (size_t i = 0; i < 5000; ++i)
    REQUIRE(randLabels[i] == revertedLabels[i]);
}

// Test structures.
class TestInner
{
 public:
  TestInner(char c, const string& s) : c(c), s(s) { }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar(CEREAL_NVP(c));
    ar(CEREAL_NVP(s));
  }

  // Public members for testing.
  char c;
  string s;
};

class Test
{
 public:
  Test(int x, int y) : x(x), y(y), ina('a', "hello"), inb('b', "goodbye") { }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar(CEREAL_NVP(x));
    ar(CEREAL_NVP(y));
    ar(CEREAL_NVP(ina));
    ar(CEREAL_NVP(inb));
  }

  // Public members for testing.
  int x;
  int y;
  TestInner ina;
  TestInner inb;
};

/**
 * Make sure we can load and save.
 *
 * Make sure to remove this one when releasing mlpack 5.0.0
 */
TEST_CASE("LoadBinaryTest", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);

  REQUIRE(Save("test.bin", x, NoFatal + BIN) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(Load("test.bin", y, NoFatal + BIN) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);
}

TEST_CASE("LoadBinaryTestOptions", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);

  DataOptions opts;
  opts.Format() = FileType::BIN;
  opts.Fatal() = false;

  REQUIRE(Save("test.bin", x, opts) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(Load("test.bin", y, opts) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);
}

TEST_CASE("LoadBinaryTestInOptions", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);

  REQUIRE(Save("test.bin", x, NoFatal + BIN) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(Load("test.bin", y, NoFatal + BIN) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);
}

TEST_CASE("LoadAutoDetectTestInOptions", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);

  REQUIRE(Save("test.bin", x, NoFatal + AutoDetect) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(Load("test.bin", y, NoFatal + AutoDetect) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);

  REQUIRE(Save("test.xml", x, NoFatal + AutoDetect) == true);
  REQUIRE(Load("test.xml", y, NoFatal + AutoDetect) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);

  REQUIRE(Save("test.JSON", x, NoFatal + AutoDetect) == true);
  REQUIRE(Load("test.JSON", y, NoFatal + AutoDetect) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);
}

TEST_CASE("LoadBinaryTestBadOptions", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);

  REQUIRE_THROWS_AS(Save("test.bin", x, NoFatal + BIN + CSV),
      std::invalid_argument);

  REQUIRE(Save("test.bin", x, NoFatal + BIN) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE_THROWS_AS(Load("test.bin", y, NoFatal + BIN + HDF5),
      std::invalid_argument);
}

/**
 * Make sure we can load and save.
 *
 * Make sure to remove this one when releasing mlpack 5.0.0
 */
TEST_CASE("LoadXMLTest", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);

  REQUIRE(Save("test.xml", x, NoFatal + XML) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(Load("test.xml", y, NoFatal + XML) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);
}

TEST_CASE("LoadXMLTestOptions", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);

  DataOptions opts;
  opts.Fatal() = false;
  opts.Format() = FileType::XML;
  REQUIRE(Save("test.xml", x, opts) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(Load("test.xml", y, opts) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);
}

TEST_CASE("LoadXMLTestInOptions", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);

  REQUIRE(Save("test.xml", x, NoFatal + XML) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(Load("test.xml", y, NoFatal + XML) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);
}

/**
 * Make sure we can load and save.
 */
TEST_CASE("LoadJsonTestOptions", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);
  DataOptions opts;
  opts.Fatal() = false;
  opts.Format() = FileType::JSON;

  REQUIRE(Save("test.json", x, opts) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(Load("test.json", y, opts) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);
}

TEST_CASE("LoadJsonTestInOptions", "[LoadSaveTest][tiny]")
{
  Test x(10, 12);

  REQUIRE(Save("test.json", x, NoFatal + JSON) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(Load("test.json", y, NoFatal + JSON) == true);

  REQUIRE(y.x == x.x);
  REQUIRE(y.y == x.y);
  REQUIRE(y.ina.c == x.ina.c);
  REQUIRE(y.ina.s == x.ina.s);
  REQUIRE(y.inb.c == x.inb.c);
  REQUIRE(y.inb.s == x.inb.s);
}

/**
 * Test DatasetInfo by making a map for a dimension.
 */
TEST_CASE("DatasetInfoTest", "[LoadSaveTest][tiny]")
{
  DatasetInfo di(100);

  // Do all types default to numeric?
  for (size_t i = 0; i < 100; ++i)
  {
    REQUIRE(di.Type(i) == Datatype::numeric);
    REQUIRE(di.NumMappings(i) == 0);
  }

  // Okay.  Add some mappings for dimension 3.
  const size_t first = di.MapString<size_t>("test_mapping_1", 3);
  const size_t second = di.MapString<size_t>("test_mapping_2", 3);
  const size_t third = di.MapString<size_t>("test_mapping_3", 3);

  REQUIRE(first == 0);
  REQUIRE(second == 1);
  REQUIRE(third == 2);

  // Now dimension 3 should be categorical.
  for (size_t i = 0; i < 100; ++i)
  {
    if (i == 3)
    {
      REQUIRE(di.Type(i) == Datatype::categorical);
      REQUIRE(di.NumMappings(i) == 3);
    }
    else
    {
      REQUIRE(di.Type(i) == Datatype::numeric);
      REQUIRE(di.NumMappings(i) == 0);
    }
  }

  // Get the mappings back.
  const string& strFirst = di.UnmapString(first, 3);
  const string& strSecond = di.UnmapString(second, 3);
  const string& strThird = di.UnmapString(third, 3);

  REQUIRE(strFirst == "test_mapping_1");
  REQUIRE(strSecond == "test_mapping_2");
  REQUIRE(strThird == "test_mapping_3");
}

/**
 * Test loading regular CSV with DatasetInfo.  Everything should be numeric.
 */
TEST_CASE("RegularCSVDatasetInfoLoad", "[LoadSaveTest][tiny]")
{
  vector<string> testFiles;
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
    TextOptions opts = Categorical;

    if (!Load(testFiles[i], one))
      FAIL("Cannot load dataset");
    if (!Load(testFiles[i], two, opts))
      FAIL("Cannot load dataset");

    // Check that the matrices contain the same information.
    REQUIRE(one.n_elem == two.n_elem);
    REQUIRE(one.n_rows == two.n_rows);
    REQUIRE(one.n_cols == two.n_cols);
    for (size_t i = 0; i < one.n_elem; ++i)
    {
      if (std::abs(one[i]) < 1e-8)
        REQUIRE(two[i] == Approx(.0).margin(1e-10));
      else
        REQUIRE(one[i] == Approx(two[i]).epsilon(1e-7));
    }

    // Check that all dimensions are numeric.
    for (size_t i = 0; i < two.n_rows; ++i)
      REQUIRE(opts.DatasetInfo().Type(i) == Datatype::numeric);
  }
}

/**
 * Test non-transposed loading of regular CSVs with DatasetInfo.  Everything
 * should be numeric.
 */
TEST_CASE("NontransposedCSVDatasetInfoLoad", "[LoadSaveTest][tiny]")
{
  vector<string> testFiles;
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
    TextOptions opts;
    opts.Fatal() = false;
    opts.NoTranspose() = true;
    opts.Categorical() = true;
    if (!Load(testFiles[i], one, NoFatal + NoTranspose))
      FAIL("Cannot load dataset");
    if (!Load(testFiles[i], two, opts))
      FAIL("Cannot load dataset");

    // Check that the matrices contain the same information.
    REQUIRE(one.n_elem == two.n_elem);
    REQUIRE(one.n_rows == two.n_rows);
    REQUIRE(one.n_cols == two.n_cols);
    for (size_t i = 0; i < one.n_elem; ++i)
    {
      if (std::abs(one[i]) < 1e-8)
        REQUIRE(two[i] == Approx(.0).margin(1e-10));
      else
        REQUIRE(one[i] == Approx(two[i]).epsilon(1e-7));
    }

    // Check that all dimensions are numeric.
    for (size_t i = 0; i < two.n_rows; ++i)
      REQUIRE(opts.DatasetInfo().Type(i) == Datatype::numeric);
  }
}

/**
 * Create a file with a categorical string feature, then load it.
 */
TEST_CASE("CategoricalCSVLoadTest00", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 2, hello" << endl;
  f << "3, 4, goodbye" << endl;
  f << "5, 6, coffee" << endl;
  f << "7, 8, confusion" << endl;
  f << "9, 10, hello" << endl;
  f << "11, 12, confusion" << endl;
  f << "13, 14, confusion" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts = Categorical;

  if (!Load("test.csv", matrix, opts))
    FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 7);
  REQUIRE(matrix.n_rows == 3);

  REQUIRE(matrix(0, 0) == 1);
  REQUIRE(matrix(1, 0) == 2);
  REQUIRE(matrix(2, 0) == 0);
  REQUIRE(matrix(0, 1) == 3);
  REQUIRE(matrix(1, 1) == 4);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(0, 2) == 5);
  REQUIRE(matrix(1, 2) == 6);
  REQUIRE(matrix(2, 2) == 2);
  REQUIRE(matrix(0, 3) == 7);
  REQUIRE(matrix(1, 3) == 8);
  REQUIRE(matrix(2, 3) == 3);
  REQUIRE(matrix(0, 4) == 9);
  REQUIRE(matrix(1, 4) == 10);
  REQUIRE(matrix(2, 4) == 0);
  REQUIRE(matrix(0, 5) == 11);
  REQUIRE(matrix(1, 5) == 12);
  REQUIRE(matrix(2, 5) == 3);
  REQUIRE(matrix(0, 6) == 13);
  REQUIRE(matrix(1, 6) == 14);
  REQUIRE(matrix(2, 6) == 3);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::categorical);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("hello", 2) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("goodbye", 2) == 1);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("coffee", 2) == 2);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("confusion", 2) == 3);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 2) == "hello");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 2) == "goodbye");
  REQUIRE(opts.DatasetInfo().UnmapString(2, 2) == "coffee");
  REQUIRE(opts.DatasetInfo().UnmapString(3, 2) == "confusion");

  remove("test.csv");
}

TEST_CASE("CategoricalCSVLoadTest01", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << " , 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts = Categorical;

  if (!Load("test.csv", matrix, opts))
    FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 4);
  REQUIRE(matrix.n_rows == 3);

  REQUIRE(matrix(0, 0) == 0);
  REQUIRE(matrix(0, 1) == 0);
  REQUIRE(matrix(0, 2) == 1);
  REQUIRE(matrix(0, 3) == 0);
  REQUIRE(matrix(1, 0) == 1);
  REQUIRE(matrix(1, 1) == 1);
  REQUIRE(matrix(1, 2) == 1);
  REQUIRE(matrix(1, 3) == 1);
  REQUIRE(matrix(2, 0) == 1);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(2, 2) == 1);
  REQUIRE(matrix(2, 3) == 1);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(3) == Datatype::numeric);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("1", 0) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("", 0) == 1);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 0) == "1");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 0) == "");

  remove("test.csv");
}

TEST_CASE("CategoricalCSVLoadTest02", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 1, 1" << endl;
  f << ", 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts = Categorical;

  if (!Load("test.csv", matrix, opts))
    FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 4);
  REQUIRE(matrix.n_rows == 3);

  REQUIRE(matrix(0, 0) == 0);
  REQUIRE(matrix(0, 1) == 1);
  REQUIRE(matrix(0, 2) == 0);
  REQUIRE(matrix(0, 3) == 0);
  REQUIRE(matrix(1, 0) == 1);
  REQUIRE(matrix(1, 1) == 1);
  REQUIRE(matrix(1, 2) == 1);
  REQUIRE(matrix(1, 3) == 1);
  REQUIRE(matrix(2, 0) == 1);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(2, 2) == 1);
  REQUIRE(matrix(2, 3) == 1);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::numeric);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("", 0) == 1);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("1", 0) == 0);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 0) == "1");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 0) == "");

  remove("test.csv");
}

TEST_CASE("CategoricalCSVLoadTest03", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << ", 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts = Categorical;

  if (!Load("test.csv", matrix, opts))
    FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 4);
  REQUIRE(matrix.n_rows == 3);

  REQUIRE(matrix(0, 0) == 0);
  REQUIRE(matrix(0, 1) == 1);
  REQUIRE(matrix(0, 2) == 1);
  REQUIRE(matrix(0, 3) == 1);
  REQUIRE(matrix(1, 0) == 1);
  REQUIRE(matrix(1, 1) == 1);
  REQUIRE(matrix(1, 2) == 1);
  REQUIRE(matrix(1, 3) == 1);
  REQUIRE(matrix(2, 0) == 1);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(2, 2) == 1);
  REQUIRE(matrix(2, 3) == 1);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::numeric);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("", 0) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("1", 0) == 1);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 0) == "");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 0) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalCSVLoadTest04", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "200-DM, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts = Categorical;

  if (!Load("test.csv", matrix, opts))
    FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 4);
  REQUIRE(matrix.n_rows == 3);

  REQUIRE(matrix(0, 0) == 0);
  REQUIRE(matrix(0, 1) == 1);
  REQUIRE(matrix(0, 2) == 1);
  REQUIRE(matrix(0, 3) == 1);
  REQUIRE(matrix(1, 0) == 1);
  REQUIRE(matrix(1, 1) == 1);
  REQUIRE(matrix(1, 2) == 1);
  REQUIRE(matrix(1, 3) == 1);
  REQUIRE(matrix(2, 0) == 1);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(2, 2) == 1);
  REQUIRE(matrix(2, 3) == 1);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::numeric);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("200-DM", 0) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("1", 0) == 1);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 0) == "200-DM");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 0) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest00", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 2, hello" << endl;
  f << "3, 4, goodbye" << endl;
  f << "5, 6, coffee" << endl;
  f << "7, 8, confusion" << endl;
  f << "9, 10, hello" << endl;
  f << "11, 12, 15" << endl;
  f << "13, 14, confusion" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts;
  opts.Categorical() = true;
  opts.NoTranspose() = true;
  opts.Fatal() = false;
  if (!Load("test.csv", matrix, opts))
      FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 3);
  REQUIRE(matrix.n_rows == 7);

  REQUIRE(matrix(0, 0) == 0);
  REQUIRE(matrix(0, 1) == 1);
  REQUIRE(matrix(0, 2) == 2);
  REQUIRE(matrix(1, 0) == 0);
  REQUIRE(matrix(1, 1) == 1);
  REQUIRE(matrix(1, 2) == 2);
  REQUIRE(matrix(2, 0) == 0);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(2, 2) == 2);
  REQUIRE(matrix(3, 0) == 0);
  REQUIRE(matrix(3, 1) == 1);
  REQUIRE(matrix(3, 2) == 2);
  REQUIRE(matrix(4, 0) == 0);
  REQUIRE(matrix(4, 1) == 1);
  REQUIRE(matrix(4, 2) == 2);
  REQUIRE(matrix(5, 0) == 11);
  REQUIRE(matrix(5, 1) == 12);
  REQUIRE(matrix(5, 2) == 15);
  REQUIRE(matrix(6, 0) == 0);
  REQUIRE(matrix(6, 1) == 1);
  REQUIRE(matrix(6, 2) == 2);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(3) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(4) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(5) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(6) == Datatype::categorical);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("1", 0) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("2", 0) == 1);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("hello", 0) == 2);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("3", 1) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("4", 1) == 1);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("goodbye", 1) == 2);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("5", 2) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("6", 2) == 1);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("coffee", 2) == 2);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("7", 3) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("8", 3) == 1);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("confusion", 3) == 2);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("9", 4) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("10", 4) == 1);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("hello", 4) == 2);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("13", 6) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("14", 6) == 1);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("confusion", 6) == 2);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 0) == "1");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 0) == "2");
  REQUIRE(opts.DatasetInfo().UnmapString(2, 0) == "hello");
  REQUIRE(opts.DatasetInfo().UnmapString(0, 1) == "3");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 1) == "4");
  REQUIRE(opts.DatasetInfo().UnmapString(2, 1) == "goodbye");
  REQUIRE(opts.DatasetInfo().UnmapString(0, 2) == "5");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 2) == "6");
  REQUIRE(opts.DatasetInfo().UnmapString(2, 2) == "coffee");
  REQUIRE(opts.DatasetInfo().UnmapString(0, 3) == "7");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 3) == "8");
  REQUIRE(opts.DatasetInfo().UnmapString(2, 3) == "confusion");
  REQUIRE(opts.DatasetInfo().UnmapString(0, 4) == "9");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 4) == "10");
  REQUIRE(opts.DatasetInfo().UnmapString(2, 4) == "hello");
  REQUIRE(opts.DatasetInfo().UnmapString(0, 6) == "13");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 6) == "14");
  REQUIRE(opts.DatasetInfo().UnmapString(2, 6) == "confusion");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest01", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << " , 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts;
  opts.Categorical() = true;
  opts.NoTranspose() = true;
  opts.Fatal() = false;

  if (!Load("test.csv", matrix, opts))
      FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 3);
  REQUIRE(matrix.n_rows == 4);

  REQUIRE(matrix(0, 0) == 1);
  REQUIRE(matrix(0, 1) == 1);
  REQUIRE(matrix(0, 2) == 1);
  REQUIRE(matrix(1, 0) == 1);
  REQUIRE(matrix(1, 1) == 1);
  REQUIRE(matrix(1, 2) == 1);
  REQUIRE(matrix(2, 0) == 0);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(2, 2) == 1);
  REQUIRE(matrix(3, 0) == 1);
  REQUIRE(matrix(3, 1) == 1);
  REQUIRE(matrix(3, 2) == 1);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(3) == Datatype::numeric);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("", 2) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("1", 2) == 1);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 2) == "");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 2) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest02", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 1, 1" << endl;
  f << ", 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts;
  opts.Categorical() = true;
  opts.NoTranspose() = true;
  opts.Fatal() = false;

  if (!Load("test.csv", matrix, opts))
      FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 3);
  REQUIRE(matrix.n_rows == 4);

  REQUIRE(matrix(0, 0) == 1);
  REQUIRE(matrix(0, 1) == 1);
  REQUIRE(matrix(0, 2) == 1);
  REQUIRE(matrix(1, 0) == 0);
  REQUIRE(matrix(1, 1) == 1);
  REQUIRE(matrix(1, 2) == 1);
  REQUIRE(matrix(2, 0) == 1);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(2, 2) == 1);
  REQUIRE(matrix(3, 0) == 1);
  REQUIRE(matrix(3, 1) == 1);
  REQUIRE(matrix(3, 2) == 1);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(3) == Datatype::numeric);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("", 1) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("1", 1) == 1);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 1) == "");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 1) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest03", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << ",  1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f << "1, 1, 1" << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts;
  opts.Categorical() = true;
  opts.NoTranspose() = true;
  opts.Fatal() = false;

  if (!Load("test.csv", matrix, opts))
      FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 3);
  REQUIRE(matrix.n_rows == 4);

  REQUIRE(matrix(0, 0) == 0);
  REQUIRE(matrix(0, 1) == 1);
  REQUIRE(matrix(0, 2) == 1);
  REQUIRE(matrix(1, 0) == 1);
  REQUIRE(matrix(1, 1) == 1);
  REQUIRE(matrix(1, 2) == 1);
  REQUIRE(matrix(2, 0) == 1);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(2, 2) == 1);
  REQUIRE(matrix(3, 0) == 1);
  REQUIRE(matrix(3, 1) == 1);
  REQUIRE(matrix(3, 2) == 1);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(3) == Datatype::numeric);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("", 1) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("1", 1) == 1);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 1) == "");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 1) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest04", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << " 200-DM ,   1  , 1  " << endl;
  f << "  1 , 1  , 1  " << endl;
  f << "  1  ,   1  ,  1  " << endl;
  f << "  1  , 1  , 1  " << endl;
  f.close();

  // Load the test CSV.
  arma::umat matrix;
  TextOptions opts;
  opts.Categorical() = true;
  opts.NoTranspose() = true;
  opts.Fatal() = false;

  if (!Load("test.csv", matrix, opts))
    FAIL("Cannot load dataset");

  REQUIRE(matrix.n_cols == 3);
  REQUIRE(matrix.n_rows == 4);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(3) == Datatype::numeric);

  REQUIRE(matrix(0, 0) == 0);
  REQUIRE(matrix(0, 1) == 1);
  REQUIRE(matrix(0, 2) == 1);
  REQUIRE(matrix(1, 0) == 1);
  REQUIRE(matrix(1, 1) == 1);
  REQUIRE(matrix(1, 2) == 1);
  REQUIRE(matrix(2, 0) == 1);
  REQUIRE(matrix(2, 1) == 1);
  REQUIRE(matrix(2, 2) == 1);
  REQUIRE(matrix(3, 0) == 1);
  REQUIRE(matrix(3, 1) == 1);
  REQUIRE(matrix(3, 2) == 1);

  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("200-DM", 1) == 0);
  REQUIRE(opts.DatasetInfo().MapString<arma::uword>("1", 1) == 1);

  REQUIRE(opts.DatasetInfo().UnmapString(0, 1) == "200-DM");
  REQUIRE(opts.DatasetInfo().UnmapString(1, 1) == "1");

  remove("test.csv");
}

/**
 * A harder test CSV based on the concerns in #658.
 */
TEST_CASE("HarderKeonTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "a,, 13,\t, 0" << endl;
  f << "b, 3, 14, hello,1" << endl;
  f << "b, 4, 15, , 2" << endl;
  f << ", 5, 16, ," << endl;
  f.close();

  // Load transposed.
  arma::mat dataset;
  TextOptions opts = Categorical;

  if (!Load("test.csv", dataset, opts))
    FAIL("Cannot load dataset");

  REQUIRE(dataset.n_rows == 5);
  REQUIRE(dataset.n_cols == 4);

  REQUIRE(opts.DatasetInfo().Dimensionality() == 5);
  REQUIRE(opts.DatasetInfo().NumMappings(0) == 3);
  REQUIRE(opts.DatasetInfo().NumMappings(1) == 4);
  REQUIRE(opts.DatasetInfo().NumMappings(2) == 0);
  REQUIRE(opts.DatasetInfo().NumMappings(3) == 2); // \t and "" are equivalent.
  REQUIRE(opts.DatasetInfo().NumMappings(4) == 4);

  // Now load non-transposed.
  TextOptions ntOpts;
  ntOpts.Categorical() = true;
  ntOpts.NoTranspose() = true;
  ntOpts.Fatal() = false;

  if (!Load("test.csv", dataset, ntOpts))
    FAIL("Cannot load dataset");

  REQUIRE(dataset.n_rows == 4);
  REQUIRE(dataset.n_cols == 5);

  REQUIRE(ntOpts.DatasetInfo().Dimensionality() == 4);
  REQUIRE(ntOpts.DatasetInfo().NumMappings(0) == 4);
  REQUIRE(ntOpts.DatasetInfo().NumMappings(1) == 5);
  REQUIRE(ntOpts.DatasetInfo().NumMappings(2) == 5);
  REQUIRE(ntOpts.DatasetInfo().NumMappings(3) == 3);

  remove("test.csv");
}

/**
 * A simple ARFF load test.  Two attributes, both numeric.
 */
TEST_CASE("SimpleARFFTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.arff", fstream::out);
  f << "@relation test" << endl;
  f << endl;
  f << "@attribute one NUMERIC" << endl;
  f << "@attribute two NUMERIC" << endl;
  f << endl;
  f << "@data" << endl;
  f << "1, 2" << endl;
  f << "3, 4" << endl;
  f << "5, 6" << endl;
  f << "7, 8" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts = Categorical;

  if (!Load("test.arff", dataset, opts))
    FAIL("Cannot load dataset");

  REQUIRE(opts.DatasetInfo().Dimensionality() == 2);
  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);

  REQUIRE(dataset.n_rows == 2);
  REQUIRE(dataset.n_cols == 4);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(dataset[i] == Approx(double(i + 1)).epsilon(1e-7));

  remove("test.arff");
}

/**
 * Another simple ARFF load test.  Three attributes, two categorical, one
 * numeric.
 */
TEST_CASE("SimpleARFFCategoricalTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.arff", fstream::out);
  f << "@relation test" << endl;
  f << endl;
  f << "@attribute one STRING" << endl;
  f << "@attribute two REAL" << endl;
  f << endl;
  f << "@attribute three STRING" << endl;
  f << endl;
  f << "% a comment line " << endl;
  f << endl;
  f << "@data" << endl;
  f << "hello, 1, moo" << endl;
  f << "cheese, 2.34, goodbye" << endl;
  f << "seven, 1.03e+5, moo" << endl;
  f << "hello, -1.3, goodbye" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts = Categorical;

  if (!Load("test.arff", dataset, opts))
    FAIL("Cannot load dataset");

  REQUIRE(opts.DatasetInfo().Dimensionality() == 3);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().NumMappings(0) == 3);
  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().NumMappings(2) == 2);

  REQUIRE(dataset.n_rows == 3);
  REQUIRE(dataset.n_cols == 4);

  // The first dimension must all be different (except the ones that are the
  // same).
  REQUIRE(dataset(0, 0) == dataset(0, 3));
  REQUIRE(dataset(0, 0) != dataset(0, 1));
  REQUIRE(dataset(0, 1) != dataset(0, 2));
  REQUIRE(dataset(0, 2) != dataset(0, 0));

  REQUIRE(dataset(1, 0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(dataset(1, 1) == Approx(2.34).epsilon(1e-7));
  REQUIRE(dataset(1, 2) == Approx(1.03e5).epsilon(1e-7));
  REQUIRE(dataset(1, 3) == Approx(-1.3).epsilon(1e-7));

  REQUIRE(dataset(2, 0) == dataset(2, 2));
  REQUIRE(dataset(2, 1) == dataset(2, 3));
  REQUIRE(dataset(2, 0) != dataset(2, 1));

  remove("test.arff");
}

/**
 * A harder ARFF test, where we have each type of supported value, and some
 * random whitespace too.
 */
TEST_CASE("HarderARFFTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.arff", fstream::out);
  f << "@relation    \t test" << endl;
  f << endl;
  f << endl;
  f << "@attribute @@@@flfl numeric" << endl;
  f << endl;
  f << "% comment" << endl;
  f << "@attribute \"hello world\" string" << endl;
  f << "@attribute 12345 integer" << endl;
  f << "@attribute real real" << endl;
  f << "@attribute \"blah blah blah     \t \" numeric % comment" << endl;
  f << "% comment" << endl;
  f << "@data" << endl;
  f << "1, one, 3, 4.5, 6" << endl;
  f << "2, two, 4, 5.5, 7 % comment" << endl;
  f << "3, \"three five, six\", 5, 6.5, 8" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts = Categorical;

  if (!Load("test.arff", dataset, opts))
    FAIL("Cannot load dataset");

  REQUIRE(opts.DatasetInfo().Dimensionality() == 5);

  REQUIRE(opts.DatasetInfo().Type(0) == Datatype::numeric);

  REQUIRE(opts.DatasetInfo().Type(1) == Datatype::categorical);
  REQUIRE(opts.DatasetInfo().NumMappings(1) == 3);

  REQUIRE(opts.DatasetInfo().Type(2) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(3) == Datatype::numeric);
  REQUIRE(opts.DatasetInfo().Type(4) == Datatype::numeric);

  REQUIRE(dataset.n_rows == 5);
  REQUIRE(dataset.n_cols == 3);

  REQUIRE(dataset(0, 0) == Approx(1.0).epsilon(1e-7));
  REQUIRE(dataset(0, 1) == Approx(2.0).epsilon(1e-7));
  REQUIRE(dataset(0, 2) == Approx(3.0).epsilon(1e-7));

  REQUIRE(dataset(1, 0) != dataset(1, 1));
  REQUIRE(dataset(1, 1) != dataset(1, 2));
  REQUIRE(dataset(1, 0) != dataset(1, 2));

  REQUIRE(dataset(2, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(dataset(2, 1) == Approx(4.0).epsilon(1e-7));
  REQUIRE(dataset(2, 2) == Approx(5.0).epsilon(1e-7));

  REQUIRE(dataset(3, 0) == Approx(4.5).epsilon(1e-7));
  REQUIRE(dataset(3, 1) == Approx(5.5).epsilon(1e-7));
  REQUIRE(dataset(3, 2) == Approx(6.5).epsilon(1e-7));

  REQUIRE(dataset(4, 0) == Approx(6.0).epsilon(1e-7));
  REQUIRE(dataset(4, 1) == Approx(7.0).epsilon(1e-7));
  REQUIRE(dataset(4, 2) == Approx(8.0).epsilon(1e-7));

  remove("test.arff");
}

/**
 * If we pass a bad DatasetInfo, it should throw.
 */
TEST_CASE("BadDatasetInfoARFFTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.arff", fstream::out);
  f << "@relation    \t test" << endl;
  f << endl;
  f << endl;
  f << "@attribute @@@@flfl numeric" << endl;
  f << endl;
  f << "% comment" << endl;
  f << "@attribute \"hello world\" string" << endl;
  f << "@attribute 12345 integer" << endl;
  f << "@attribute real real" << endl;
  f << "@attribute \"blah blah blah     \t \" numeric % comment" << endl;
  f << "% comment" << endl;
  f << "@data" << endl;
  f << "1, one, 3, 4.5, 6" << endl;
  f << "2, two, 4, 5.5, 7 % comment" << endl;
  f << "3, \"three five, six\", 5, 6.5, 8" << endl;
  f.close();

  arma::mat dataset;
  DatasetInfo info(6);

  REQUIRE_THROWS(LoadARFF("test.arff", dataset, info, true));

  remove("test.arff");
}

/**
 * If file is not found, it should throw.
 */
TEST_CASE("NonExistentFileARFFTest", "[LoadSaveTest][tiny]")
{
  arma::mat dataset;
  DatasetInfo info;

  REQUIRE_THROWS(LoadARFF("nonexistentfile.arff", dataset, info, true));
}

/**
 * A test to check whether the arff loader is case insensitive to declarations:
 * @relation, @attribute, @data.
 */
TEST_CASE("CaseTest", "[LoadSaveTest][tiny]")
{
  arma::mat dataset;

  DatasetMapper<IncrementPolicy> info;

  LoadARFF<double, IncrementPolicy>("casecheck.arff", dataset, info, true);

  REQUIRE(dataset.n_rows == 2);
  REQUIRE(dataset.n_cols == 3);
}

/**
 * Ensure that a failure happens if we set a category to use capital letters but
 * it receives them in lowercase.
 */
TEST_CASE("CategoryCaseTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.arff", fstream::out);
  f << "@relation    \t test" << endl;
  f << endl;
  f << endl;
  f << "@attribute @@@@flfl {A, B, C, D}" << endl;
  f << endl;
  f << "% comment" << endl;
  f << "@attribute \"hello world\" string" << endl;
  f << "@attribute 12345 integer" << endl;
  f << "@attribute real real" << endl;
  f << "@attribute \"blah blah blah     \t \" numeric % comment" << endl;
  f << "% comment" << endl;
  f << "@data" << endl;
  f << "A, one, 3, 4.5, 6" << endl;
  f << "B, two, 4, 5.5, 7 % comment" << endl;
  f << "c, \"three five, six\", 5, 6.5, 8" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts;
  opts.Categorical() = true;
  opts.NoTranspose() = false;
  opts.Fatal() = true;

  // Make sure to parse with fatal errors (that's what the `true` parameter
  // means).
  REQUIRE_THROWS_AS(Load("test.arff", dataset, opts),
      std::runtime_error);

  remove("test.arff");
}

/**
 * Test that a CSV with the wrong number of columns fails.
 */
TEST_CASE("MalformedCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7" << endl;
  f << "8, 9, 10, 11" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts = Categorical;

  REQUIRE(!Load("test.csv", dataset, opts));

  remove("test.csv");
}

/**
 * Test that a TSV can load with LoadCSV.
 */
TEST_CASE("LoadCSVTSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.tsv", fstream::out);
  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts = Categorical;

  REQUIRE(Load("test.tsv", dataset, opts));

  REQUIRE(dataset.n_cols == 2);
  REQUIRE(dataset.n_rows == 4);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(dataset[i] == i + 1);

  remove("test.tsv");
}

/**
 * Test that a text file can load with LoadCSV.
 */
TEST_CASE("LoadCSVTXTTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.txt", fstream::out);
  f << "1 2 3 4" << endl;
  f << "5 6 7 8" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts = Categorical;

  REQUIRE(Load("test.txt", dataset, opts));

  REQUIRE(dataset.n_cols == 2);
  REQUIRE(dataset.n_rows == 4);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(dataset[i] == i + 1);

  remove("test.txt");
}

/**
 * Test that a non-transposed CSV with the wrong number of columns fails.
 */
TEST_CASE("MalformedNoTransposeCSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7" << endl;
  f << "8, 9, 10, 11" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts;
  opts.Categorical() = true;
  opts.NoTranspose() = true;
  opts.Fatal() = false;

  REQUIRE(!Load("test.csv", dataset, opts));

  remove("test.csv");
}

/**
 * Test that a non-transposed TSV can load with LoadCSV.
 */
TEST_CASE("LoadCSVNoTransposeTSVTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.tsv", fstream::out);
  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts;
  opts.Categorical() = true;
  opts.NoTranspose() = true;
  opts.Fatal() = false;

  REQUIRE(Load("test.tsv", dataset, opts));

  REQUIRE(dataset.n_cols == 4);
  REQUIRE(dataset.n_rows == 2);

  REQUIRE(dataset[0] == 1);
  REQUIRE(dataset[1] == 5);
  REQUIRE(dataset[2] == 2);
  REQUIRE(dataset[3] == 6);
  REQUIRE(dataset[4] == 3);
  REQUIRE(dataset[5] == 7);
  REQUIRE(dataset[6] == 4);
  REQUIRE(dataset[7] == 8);

  remove("test.tsv");
}

/**
 * Test that a non-transposed text file can load with LoadCSV.
 */
TEST_CASE("LoadCSVNoTransposeTXTTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.txt", fstream::out);
  f << "1 2 3 4" << endl;
  f << "5 6 7 8" << endl;
  f.close();

  arma::mat dataset;
  TextOptions opts;
  opts.Categorical() = true;
  opts.NoTranspose() = true;
  opts.Fatal() = false;

  REQUIRE(Load("test.txt", dataset, opts));

  REQUIRE(dataset.n_cols == 4);
  REQUIRE(dataset.n_rows == 2);

  REQUIRE(dataset[0] == 1);
  REQUIRE(dataset[1] == 5);
  REQUIRE(dataset[2] == 2);
  REQUIRE(dataset[3] == 6);
  REQUIRE(dataset[4] == 3);
  REQUIRE(dataset[5] == 7);
  REQUIRE(dataset[6] == 4);
  REQUIRE(dataset[7] == 8);

  remove("test.txt");
}

/**
 * Make sure if we load a CSV with a header, that that header doesn't get loaded
 * as a point.
 */
TEST_CASE("LoadCSVHeaderTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "a,b,c,d" << endl;
  f << "1,2,3,4" << endl;
  f << "5,6,7,8" << endl;

  arma::mat dataset;
  TextOptions opts;
  opts.HasHeaders() = true;
  Load("test.csv", dataset, opts);

  arma::field<std::string> headers = opts.Headers();

  REQUIRE(dataset.n_rows == 4);
  REQUIRE(dataset.n_cols == 2);
  REQUIRE(headers.n_elem == 4);
  REQUIRE(headers.at(0) == "a");
  REQUIRE(headers.at(1) == "b");
  REQUIRE(headers.at(2) == "c");
  REQUIRE(headers.at(3) == "d");
}

TEST_CASE("DataOptionsTest", "[LoadSaveTest][tiny]")
{
  DataOptions opts1, opts2, opts3;

  opts1.Fatal() = false;
  opts2.Fatal() = false;
  opts1.Format() = FileType::FileTypeUnknown;
  opts2.Format() = FileType::CSVASCII;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.Format() == FileType::CSVASCII);

  opts1.Fatal() = true;
  opts2.Fatal() = true;
  opts1.Format() = FileType::AutoDetect;
  opts2.Format() = FileType::RawASCII;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.Format() == FileType::RawASCII);

  opts2.Fatal() = true;
  opts1.Format() = FileType::RawASCII;
  opts2.Format() = FileType::AutoDetect;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.Format() == FileType::RawASCII);

  opts1.Format() = FileType::CSVASCII;
  opts2.Format() = FileType::FileTypeUnknown;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Format() == FileType::CSVASCII);
}

TEST_CASE("MatrixOptionsTest", "[LoadSaveTest][tiny]")
{
  MatrixOptions opts1, opts2, opts3;

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  opts2.Fatal() = false;
  opts2.NoTranspose() = false;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts2.Fatal() = true;
  opts2.NoTranspose() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);

  opts1.Fatal() = true;
  opts2.Fatal() = false;
  opts1.NoTranspose() = true;
  opts2.NoTranspose() = false;
  REQUIRE_THROWS_AS(opts3 = opts1 + opts2, std::invalid_argument);

  opts1.Fatal() = false;
  opts2.Fatal() = true;
  opts1.NoTranspose() = false;
  opts2.NoTranspose() = true;
  REQUIRE_THROWS_AS(opts3 = opts1 + opts2, std::invalid_argument);
}

TEST_CASE("TextOptionsTest", "[LoadSaveTest][tiny]")
{
  TextOptions opts1, opts2, opts3;

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  opts1.Categorical() = false;
  opts1.HasHeaders() = false;
  opts1.MissingToNan() = false;
  opts1.Semicolon() = false;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts1.Categorical() = true;
  opts1.HasHeaders() = true;
  opts1.MissingToNan() = true;
  opts1.Semicolon() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  opts1.Categorical() = false;
  opts1.HasHeaders() = false;
  opts1.MissingToNan() = false;
  opts1.Semicolon() = false;
  opts2.Fatal() = false;
  opts2.NoTranspose() = false;
  opts2.Categorical() = false;
  opts2.HasHeaders() = false;
  opts2.MissingToNan() = false;
  opts2.Semicolon() = false;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts1.Categorical() = true;
  opts1.HasHeaders() = true;
  opts1.MissingToNan() = true;
  opts1.Semicolon() = true;
  opts2.Fatal() = true;
  opts2.NoTranspose() = true;
  opts2.Categorical() = true;
  opts2.HasHeaders() = true;
  opts2.MissingToNan() = true;
  opts2.Semicolon() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts1.Categorical() = true;
  opts1.HasHeaders() = true;
  opts1.MissingToNan() = true;
  opts1.Semicolon() = true;
  opts2.Fatal() = false;
  opts2.NoTranspose() = false;
  opts2.Categorical() = false;
  opts2.HasHeaders() = false;
  opts2.MissingToNan() = false;
  opts2.Semicolon() = false;

  REQUIRE_THROWS_AS(opts3 = opts1 + opts2, std::invalid_argument);
}

TEST_CASE("MatrixDataOptionsTest", "[LoadSaveTest][tiny]")
{
  MatrixOptions opts1;
  DataOptions opts2;

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  auto opts3 = opts1 + opts2;
  static_assert(std::is_same_v<decltype(opts3), MatrixOptions>);
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  opts2.Fatal() = false;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts2.Fatal() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);

  opts1.Fatal() = true;
  opts2.Fatal() = false;
  REQUIRE_THROWS_AS(opts3 = opts1 + opts2, std::invalid_argument);

  opts1.Fatal() = false;
  opts2.Fatal() = true;
  REQUIRE_THROWS_AS(opts3 = opts1 + opts2, std::invalid_argument);
}

TEST_CASE("TextDataOptionsTest", "[LoadSaveTest][tiny]")
{
  TextOptions opts1;
  DataOptions opts2;

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  opts1.Categorical() = false;
  opts1.HasHeaders() = false;
  opts1.MissingToNan() = false;
  opts1.Semicolon() = false;
  auto opts3 = opts1 + opts2;
  static_assert(std::is_same_v<decltype(opts3), TextOptions>);
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts1.Categorical() = true;
  opts1.HasHeaders() = true;
  opts1.MissingToNan() = true;
  opts1.Semicolon() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  opts1.Categorical() = false;
  opts1.HasHeaders() = false;
  opts1.MissingToNan() = false;
  opts1.Semicolon() = false;
  opts2.Fatal() = false;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts1.Categorical() = true;
  opts1.HasHeaders() = true;
  opts1.MissingToNan() = true;
  opts1.Semicolon() = true;
  opts2.Fatal() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);
}

TEST_CASE("MatrixTextOptionsTest", "[LoadSaveTest][tiny]")
{
  TextOptions opts1;
  MatrixOptions opts2;

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  opts1.Categorical() = false;
  opts1.HasHeaders() = false;
  opts1.MissingToNan() = false;
  opts1.Semicolon() = false;
  auto opts3 = opts1 + opts2;
  static_assert(std::is_same_v<decltype(opts3), TextOptions>);
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts1.Categorical() = true;
  opts1.HasHeaders() = true;
  opts1.MissingToNan() = true;
  opts1.Semicolon() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);

  opts1.Fatal() = false;
  opts1.NoTranspose() = false;
  opts1.Categorical() = false;
  opts1.HasHeaders() = false;
  opts1.MissingToNan() = false;
  opts1.Semicolon() = false;
  opts2.NoTranspose() = false;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == false);
  REQUIRE(opts3.NoTranspose() == false);
  REQUIRE(opts3.Categorical() == false);
  REQUIRE(opts3.HasHeaders() == false);
  REQUIRE(opts3.MissingToNan() == false);
  REQUIRE(opts3.Semicolon() == false);

  opts1.Fatal() = true;
  opts1.NoTranspose() = true;
  opts1.Categorical() = true;
  opts1.HasHeaders() = true;
  opts1.MissingToNan() = true;
  opts1.Semicolon() = true;
  opts2.NoTranspose() = true;
  opts3 = opts1 + opts2;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);
  opts3 = opts2 + opts1;
  REQUIRE(opts3.Fatal() == true);
  REQUIRE(opts3.NoTranspose() == true);
  REQUIRE(opts3.Categorical() == true);
  REQUIRE(opts3.HasHeaders() == true);
  REQUIRE(opts3.MissingToNan() == true);
  REQUIRE(opts3.Semicolon() == true);
}

// These tests only work with Armadillo 12, as we need the `strict` option to be
// available in Armadillo.
#if ARMA_VERSION_MAJOR >= 12

TEST_CASE("LoadCSVNoHeaderTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "a,b,c,d" << endl;
  f << "1,2,3,4" << endl;
  f << "5,6,7,8" << endl;

  arma::mat dataset;
  TextOptions opts;
  opts.HasHeaders() = false;
  opts.MissingToNan() = true;
  REQUIRE(Load("test.csv", dataset, opts) == true);

  REQUIRE(dataset.n_rows == 4);
  REQUIRE(dataset.n_cols == 3);
  REQUIRE(std::isnan(dataset.at(0, 0)) == true);
  REQUIRE(std::isnan(dataset.at(1, 0)) == true);
  REQUIRE(std::isnan(dataset.at(2, 0)) == true);
  REQUIRE(std::isnan(dataset.at(3, 0)) == true);
}

#endif

TEST_CASE("LoadVectorCSVFiles", "[LoadSaveTest][tiny]")
{
  std::vector<std::string> files = {"f0.csv", "f1.csv", "f2.csv", "f3.csv",
      "f4.csv", "f5.csv", "f6.csv", "f7.csv", "f8.csv", "f9.csv"};

  arma::mat dataset;
  TextOptions opts;
  REQUIRE(Load(files, dataset, opts) == true);

  REQUIRE(dataset.n_rows == 5);
  REQUIRE(dataset.n_cols == 100);
  REQUIRE(dataset(0, 10) == 1.0);
}

TEST_CASE("LoadVectorCSVOneFile", "[LoadSaveTest][tiny]")
{
  std::vector<std::string> files = {"f0.csv"};

  arma::mat dataset;
  TextOptions opts;
  REQUIRE(Load(files, dataset, opts) == true);

  REQUIRE(dataset.n_rows == 5);
  REQUIRE(dataset.n_cols == 10);
}

TEST_CASE("LoadVectorCSVEmptyFile", "[LoadSaveTest][tiny]")
{
  std::vector<std::string> files;

  arma::mat dataset;
  TextOptions opts;
  opts.Fatal() = false;
  REQUIRE(Load(files, dataset, opts) == false);
}

TEST_CASE("LoadVectorCSVDiffCols", "[LoadSaveTest][tiny]")
{
  std::vector<std::string> files = {"f0.csv", "f10.csv"};

  arma::mat dataset;
  TextOptions opts;
  REQUIRE(Load(files, dataset, opts) == false);
}

TEST_CASE("LoadVectorCSVDiffHeaders", "[LoadSaveTest][tiny]")
{
  std::vector<std::string> files = {"f0header.csv", "f10header.csv"};

  arma::mat dataset;
  TextOptions opts;
  opts.HasHeaders() = true;
  REQUIRE(Load(files, dataset, opts) == false);
}

// These tests only work with Armadillo 12, as we need the `strict` option to be
// available in Armadillo.
#if ARMA_VERSION_MAJOR >= 12

TEST_CASE("LoadVectorCSVDiffNoHeaders", "[LoadSaveTest][tiny]")
{
  std::vector<std::string> files = {"f0header.csv", "f10header.csv"};

  arma::mat dataset;
  TextOptions opts;
  opts.HasHeaders() = false;
  opts.MissingToNan() = true;
  REQUIRE(Load(files, dataset, opts) == true);

  REQUIRE(dataset.n_rows == 5);
  REQUIRE(dataset.n_cols == 22);
  REQUIRE(std::isnan(dataset.at(0, 0)) == true);
  REQUIRE(std::isnan(dataset.at(1, 0)) == true);
  REQUIRE(std::isnan(dataset.at(2, 0)) == true);
  REQUIRE(std::isnan(dataset.at(3, 0)) == true);
  // Check the Nan from the second file
  REQUIRE(std::isnan(dataset.at(0, 11)) == true);
  REQUIRE(std::isnan(dataset.at(1, 11)) == true);
  REQUIRE(std::isnan(dataset.at(2, 11)) == true);
  REQUIRE(std::isnan(dataset.at(3, 11)) == true);
}

#endif

TEST_CASE("LoadVectorCSVFilesNoTranspose", "[LoadSaveTest][tiny]")
{
  std::vector<std::string> files = {"f0.csv", "f1.csv", "f2.csv", "f3.csv",
      "f4.csv", "f5.csv", "f6.csv", "f7.csv", "f8.csv", "f9.csv"};

  arma::mat dataset;
  TextOptions opts;
  opts.NoTranspose() = true;
  opts.Fatal() = true;
  REQUIRE(Load(files, dataset, opts) == true);

  REQUIRE(dataset.n_rows == 100);
  REQUIRE(dataset.n_cols == 5);
}

// These tests only work with Armadillo 12, as we need the `strict` option to be
// available in Armadillo.
#if ARMA_VERSION_MAJOR >= 12

TEST_CASE("LoadCSVMissingNanTest", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, , 3, 4" << std::endl;
  f << "5, 6, 7, 8" << std::endl;
  f << "9, 10, 11, 12" << std::endl;

  arma::mat dataset;
  TextOptions opts;
  opts.Fatal() = false;
  opts.NoTranspose() = true;
  opts.MissingToNan() = true;

  Load("test.csv", dataset, opts);

  REQUIRE(dataset.n_rows == 3);
  REQUIRE(dataset.n_cols == 4);
  REQUIRE(std::isnan(dataset.at(0, 1)) == true);

  remove("test.csv");
}

TEST_CASE("LoadCSVMissingNanTestTransposed", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, , 3, 4" << std::endl;
  f << "5, 6, 7, 8" << std::endl;
  f << "9, 10, 11, 12" << std::endl;

  arma::mat dataset;
  TextOptions opts;
  opts.Fatal() = false;
  opts.NoTranspose() = false;
  opts.MissingToNan() = true;

  Load("test.csv", dataset, opts);

  REQUIRE(dataset.n_rows == 4);
  REQUIRE(dataset.n_cols == 3);
  REQUIRE(std::isnan(dataset.at(1, 0)) == true);

  remove("test.csv");
}

TEST_CASE("LoadCSVMissingNanTestTransposedInOptions", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, , 3, 4" << std::endl;
  f << "5, 6, 7, 8" << std::endl;
  f << "9, 10, 11, 12" << std::endl;

  arma::mat dataset;

  Load("test.csv", dataset, MissingToNan);

  REQUIRE(dataset.n_rows == 4);
  REQUIRE(dataset.n_cols == 3);
  REQUIRE(std::isnan(dataset.at(1, 0)) == true);

  remove("test.csv");
}

#endif

TEST_CASE("LoadCSVSemicolon", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1; 2; 3; 4" << std::endl;
  f << "5; 6; 7; 8" << std::endl;
  f << "9; 10; 11; 12" << std::endl;

  arma::mat dataset;
  TextOptions opts;
  opts.Fatal() = false;
  opts.NoTranspose() = false;
  opts.Semicolon() = true;

  Load("test.csv", dataset, opts);

  REQUIRE(dataset.n_rows == 4);
  REQUIRE(dataset.n_cols == 3);

  remove("test.csv");
}

TEST_CASE("LoadCSVSemicolonInOptions", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1; 2; 3; 4" << std::endl;
  f << "5; 6; 7; 8" << std::endl;
  f << "9; 10; 11; 12" << std::endl;

  arma::mat dataset;

  Load("test.csv", dataset, Semicolon);

  REQUIRE(dataset.n_rows == 4);
  REQUIRE(dataset.n_cols == 3);

  remove("test.csv");
}

TEST_CASE("LoadCSVSemicolonHeader", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "a;b;c;d" << std::endl;
  f << "1;2;3;4" << std::endl;
  f << "5;6;7;8" << std::endl;

  arma::mat dataset;
  TextOptions opts;
  opts.Fatal() = true;
  opts.NoTranspose() = false;
  opts.Semicolon() = true;
  opts.HasHeaders() = true;

  Load("test.csv", dataset, opts);

  arma::field<std::string> headers = opts.Headers();

  REQUIRE(dataset.n_rows == 4);
  REQUIRE(dataset.n_cols == 2);
  REQUIRE(headers.at(0) == "a");
  REQUIRE(headers.at(1) == "b");
  REQUIRE(headers.at(2) == "c");
  REQUIRE(headers.at(3) == "d");

  remove("test.csv");
}

#if ARMA_VERSION_MAJOR >= 12

TEST_CASE("LoadCSVSemicolonMissingToNanHeader", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "a;b;c;d" << std::endl;
  f << ";;3;4" << std::endl;
  f << "5;6;7;8" << std::endl;

  arma::mat dataset;
  TextOptions opts;
  opts.Fatal() = false;
  opts.NoTranspose() = true;
  opts.Semicolon() = true;
  opts.HasHeaders() = true;
  opts.MissingToNan() = true;

  Load("test.csv", dataset, opts);

  arma::field<std::string> headers = opts.Headers();

  REQUIRE(dataset.n_rows == 2);
  REQUIRE(dataset.n_cols == 4);
  REQUIRE(headers.at(0) == "a");
  REQUIRE(headers.at(1) == "b");
  REQUIRE(headers.at(2) == "c");
  REQUIRE(headers.at(3) == "d");
  REQUIRE(std::isnan(dataset.at(0, 0)) == true);
  REQUIRE(std::isnan(dataset.at(0, 1)) == true);
  remove("test.csv");
}

TEST_CASE("LoadCSVSemicolonMissingToNanHeaderInOptions", "[LoadSaveTest][tiny]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "a;b;c;d" << std::endl;
  f << ";;3;4" << std::endl;
  f << "5;6;7;8" << std::endl;

  arma::mat dataset;

  Load("test.csv", dataset,
      NoFatal + NoTranspose + Semicolon + HasHeaders + MissingToNan);

  REQUIRE(dataset.n_rows == 2);
  REQUIRE(dataset.n_cols == 4);
  REQUIRE(std::isnan(dataset.at(0, 0)) == true);
  REQUIRE(std::isnan(dataset.at(0, 1)) == true);
  remove("test.csv");
}

TEST_CASE("DownLoadFileOnlyAndLoad", "[LoadSaveTest]")
{
  arma::mat dataset;
  REQUIRE(Load("http://datasets.mlpack.org/iris.csv",
        dataset, Fatal + Transpose) == true);
}

TEST_CASE("DownLoadWrongURL", "[LoadSaveTest]")
{
  arma::mat dataset;
  REQUIRE(Load("http://datasets.mlpack.org/iris.csv",
        dataset, NoFatal + Transpose) == false);
}

#endif
