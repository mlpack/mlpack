/**
 * @file tests/load_save_test.cpp
 * @author Ryan Curtin
 *
 * Tests for data::Load() and data::Save().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <sstream>

#include <mlpack/core.hpp>
#include <mlpack/core/data/load_arff.hpp>
#include <mlpack/core/data/map_policies/missing_policy.hpp>
#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

/**
 * Make sure failure occurs when no extension given.
 */
TEST_CASE("NoExtensionLoad", "[LoadSaveTest]")
{
  arma::mat out;
  REQUIRE(data::Load("noextension", out) == false);
}

/**
 * Make sure failure occurs when no extension given.
 */
TEST_CASE("NoExtensionSave", "[LoadSaveTest]")
{
  arma::mat out;
  REQUIRE(data::Save("noextension", out) == false);
}

/**
 * Make sure load fails if the file does not exist.
 */
TEST_CASE("NotExistLoad", "[LoadSaveTest]")
{
  arma::mat out;
  REQUIRE(data::Load("nonexistentfile_______________.csv", out) == false);
}

/**
 * Make sure load fails if the file extension is wrong in automatic detection mode.
 */
TEST_CASE("WrongExtensionWrongLoad", "[LoadSaveTest]")
{
  // Try to load arma::arma_binary file with ".csv" extension
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.quiet_save("test_file.csv", arma::arma_binary) == true);

  // Now reload through our interface.
  REQUIRE(data::Load("test_file.csv", test) == false);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure load is successful even if the file extension is wrong when file type is specified.
 */
TEST_CASE("WrongExtensionCorrectLoad", "[LoadSaveTest]")
{
  // Try to load arma::arma_binary file with ".csv" extension
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.quiet_save("test_file.csv", arma::arma_binary) == true);

  // Now reload through our interface.
  REQUIRE(
      data::Load("test_file.csv", test, false, true, arma::arma_binary)
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
TEST_CASE("LoadCSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.csv", test) == true);

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
TEST_CASE("LoadSparseTSVTest", "[LoadSaveTest]")
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

  REQUIRE(data::Load(
      "test_sparse_file.tsv", test, true, false) == true);

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
TEST_CASE("LoadSparseTXTTest", "[LoadSaveTest]")
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

  REQUIRE(data::Load("test_sparse_file.txt", test, true, false) == true);

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
  remove("test_sparse_file.txt");
}

/**
 * Make sure a TSV is loaded correctly.
 */
TEST_CASE("LoadTSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.csv", test) == true);

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
TEST_CASE("LoadTSVExtensionTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.tsv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.tsv", test) == true);

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
TEST_CASE("LoadAnyExtensionFileTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.blah", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.blah", test, false, true, arma::raw_ascii));

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
TEST_CASE("SaveCSVTest", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(data::Save("test_file.csv", test) == true);

  // Load it in and make sure it is the same.
  arma::mat test2;
  REQUIRE(data::Load("test_file.csv", test2) == true);

  REQUIRE(test2.n_rows == 4);
  REQUIRE(test2.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test2[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure a TSV is saved correctly for a sparse matrix
 */
TEST_CASE("SaveSparseTSVTest", "[LoadSaveTest]")
{
  arma::sp_mat test = "0.1\t0\t0\t0;"
                      "0\t0.2\t0\t0;"
                      "0\t0\t0.3\t0;"
                      "0\t0\t0\t0.4;";

  REQUIRE(data::Save("test_sparse_file.tsv", test, true, false) == true);

  // Load it in and make sure it is the same.
  arma::sp_mat test2;
  REQUIRE(data::Load("test_sparse_file.tsv", test2, true, false) == true);

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
  remove("test_sparse_file.tsv");
}

/**
 * Make sure a TSV is saved correctly for a sparse matrix
 */
TEST_CASE("SaveSparseTXTTest", "[LoadSaveTest]")
{
  arma::sp_mat test = "0.1 0 0 0;"
                      "0 0.2 0 0;"
                      "0 0 0.3 0;"
                      "0 0 0 0.4;";

  REQUIRE(data::Save("test_sparse_file.txt", test, true, true) == true);

  // Load it in and make sure it is the same.
  arma::sp_mat test2;
  REQUIRE(data::Load("test_sparse_file.txt", test2, true, true) == true);

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
TEST_CASE("SaveSparseBinaryTest", "[LoadSaveTest]")
{
  arma::sp_mat test = "0.1 0 0 0;"
                      "0 0.2 0 0;"
                      "0 0 0.3 0;"
                      "0 0 0 0.4;";

  REQUIRE(data::Save("test_sparse_file.bin", test, true, false) == true);

  // Load it in and make sure it is the same.
  arma::sp_mat test2;
  REQUIRE(data::Load("test_sparse_file.bin", test2, true, false) == true);

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
  remove("test_sparse_file.bin");
}

/**
 * Make sure CSVs can be loaded in transposed form.
 */
TEST_CASE("LoadTransposedCSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.csv", test, false, true) == true);

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
TEST_CASE("LoadColVecCSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  for (size_t i = 0; i < 8; ++i)
    f << i << endl;

  f.close();

  arma::colvec test;
  REQUIRE(data::Load("test_file.csv", test, false) == true);

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
TEST_CASE("LoadColVecTransposedCSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  for (size_t i = 0; i < 8; ++i)
    f << i << ", ";
  f << "8" << endl;
  f.close();

  arma::colvec test;
  REQUIRE(data::Load("test_file.csv", test, false) == true);

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
TEST_CASE("LoadQuotedStringInCSVTest", "[LoadSaveTest]")
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
  data::DatasetInfo info;
  REQUIRE(data::Load("test_file.csv", test, info, false, true) == true);

  REQUIRE(test.n_rows == 3);
  REQUIRE(test.n_cols == 5);
  REQUIRE(info.Dimensionality() == 3);

  // Check each element for equality/ closeness.
  for (size_t i = 0; i < 5; ++i)
    REQUIRE(test.at(0, i) == Approx((double) (i + 1)).epsilon(1e-7));

  for (size_t i = 0; i < 5; ++i)
    REQUIRE(info.UnmapString(test.at(1, i), 1, 0) == elements[i]);

  for (size_t i = 0; i < 5; ++i)
    REQUIRE(info.UnmapString(test.at(2, i), 2, 0) == "field 3");

  // Clear the vector to free the space.
  elements.clear();
  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure besides numeric data "quoted strings" or
 * 'quoted strings' in txt files are loaded correctly.
 */
TEST_CASE("LoadQuotedStringInTXTTest", "[LoadSaveTest]")
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
  data::DatasetInfo info;
  REQUIRE(data::Load("test_file.txt", test, info, false, true) == true);

  REQUIRE(test.n_rows == 3);
  REQUIRE(test.n_cols == 2);
  REQUIRE(info.Dimensionality() == 3);

  // Check each element for equality/ closeness.
  for (size_t i = 0; i < 2; ++i)
    REQUIRE(test.at(0, i) == Approx((double) (i + 1)).epsilon(1e-7));

  for (size_t i = 0; i < 2; ++i)
    REQUIRE(info.UnmapString(test.at(1, i), 1, 0) == elements[i]);

  for (size_t i = 0; i < 2; ++i)
    REQUIRE(info.UnmapString(test.at(2, i), 2, 0) == "field3");

  // Clear the vector to free the space.
  elements.clear();
  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure besides numeric data "quoted strings" or
 * 'quoted strings' in tsv files are loaded correctly.
 */
TEST_CASE("LoadQuotedStringInTSVTest", "[LoadSaveTest]")
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
  data::DatasetInfo info;
  REQUIRE(data::Load("test_file.tsv", test, info, false, true) == true);

  REQUIRE(test.n_rows == 3);
  REQUIRE(test.n_cols == 5);
  REQUIRE(info.Dimensionality() == 3);

  // Check each element for equality/ closeness.
  for (size_t i = 0; i < 5; ++i)
    REQUIRE(test.at(0, i) == Approx((double) (i + 1)).epsilon(1e-7));

  for (size_t i = 0; i < 5; ++i)
    REQUIRE(info.UnmapString(test.at(1, i), 1, 0) == elements[i]);

  for (size_t i = 0; i < 5; ++i)
    REQUIRE(info.UnmapString(test.at(2, i), 2, 0) == "field 3");

  // Clear the vector to free the space.
  elements.clear();
  // Remove the file.
  remove("test_file.tsv");
}

/**
 * Make sure Load() throws an exception when trying to load a matrix into a
 * colvec or rowvec.
 */
TEST_CASE("LoadMatinVec", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 2" << endl;
  f << "3, 4" << endl;

  f.close();

  /**
   * Log::Fatal will be called when the matrix is not of the right size.
   */
  Log::Fatal.ignoreInput = true;
  arma::vec coltest;
  REQUIRE_THROWS_AS(data::Load("test_file.csv", coltest, true),
      std::runtime_error);

  arma::rowvec rowtest;
  REQUIRE_THROWS_AS(data::Load("test_file.csv", rowtest, true),
      std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("test_file.csv");
}

/**
 * Make sure that rowvecs can be loaded successfully.
 */
TEST_CASE("LoadRowVecCSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  for (size_t i = 0; i < 7; ++i)
    f << i << ", ";
  f << "7";
  f << endl;

  f.close();

  arma::rowvec test;
  REQUIRE(data::Load("test_file.csv", test, false) == true);

  REQUIRE(test.n_cols == 8);
  REQUIRE(test.n_rows == 1);

  for (size_t i = 0; i < 8 ; ++i)
    REQUIRE(test[i] == Approx((double) i).epsilon(1e-7));

  remove("test_file.csv");
}

/**
 * Make sure that we can load transposed row vectors.
 */
TEST_CASE("LoadRowVecTransposedCSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  for (size_t i = 0; i < 8; ++i)
    f << i << endl;

  f.close();

  arma::rowvec test;
  REQUIRE(data::Load("test_file.csv", test, false) == true);

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
TEST_CASE("LoadTransposedTSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.csv", test, false, true) == true);

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
TEST_CASE("LoadTransposedTSVExtensionTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.tsv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.tsv", test, false, true) == true);

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
TEST_CASE("LoadNonTransposedCSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 3, 5, 7" << endl;
  f << "2, 4, 6, 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.csv", test, false, false) == true);

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
TEST_CASE("SaveNonTransposedCSVTest", "[LoadSaveTest]")
{
  arma::mat test = "1 2;"
                   "3 4;"
                   "5 6;"
                   "7 8;";

  REQUIRE(data::Save("test_file.csv", test, false, false) == true);

  // Load it in and make sure it is in the same.
  arma::mat test2;
  REQUIRE(data::Load("test_file.csv", test2, false, false) == true);

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
TEST_CASE("LoadArmaASCIITest", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.save("test_file.txt", arma::arma_ascii));

  REQUIRE(data::Load("test_file.txt", test) == true);

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
TEST_CASE("SaveArmaASCIITest", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(data::Save("test_file.txt", test) == true);

  // Load it in and make sure it is the same.
  REQUIRE(data::Load("test_file.txt", test) == true);

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
TEST_CASE("LoadRawASCIITest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.txt", fstream::out);

  f << "1 2 3 4" << endl;
  f << "5 6 7 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.txt", test) == true);

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
TEST_CASE("LoadCSVTxtTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test_file.txt", fstream::out);

  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;

  f.close();

  arma::mat test;
  REQUIRE(data::Load("test_file.txt", test) == true);

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
TEST_CASE("LoadArmaBinaryTest", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.quiet_save("test_file.bin", arma::arma_binary)
      == true);

  // Now reload through our interface.
  REQUIRE(data::Load("test_file.bin", test) == true);

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
TEST_CASE("SaveArmaBinaryTest", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(data::Save("test_file.bin", test) == true);

  REQUIRE(data::Load("test_file.bin", test) == true);

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
TEST_CASE("SaveArmaBinaryArbitraryExtensionTest", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(data::Save("test_file.blerp.blah", test, false, true,
      arma::arma_binary) == true);

  REQUIRE(data::Load("test_file.blerp.blah", test, false, true,
      arma::arma_binary) == true);

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
TEST_CASE("LoadRawBinaryTest", "[LoadSaveTest]")
{
  arma::mat test = "1 2;"
                   "3 4;"
                   "5 6;"
                   "7 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.quiet_save("test_file.bin", arma::raw_binary)
      == true);

  // Now reload through our interface.
  REQUIRE(data::Load("test_file.bin", test) == true);

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
TEST_CASE("LoadPGMBinaryTest", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.quiet_save("test_file.pgm", arma::pgm_binary)
      == true);

  // Now reload through our interface.
  REQUIRE(data::Load("test_file.pgm", test) == true);

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
TEST_CASE("SavePGMBinaryTest", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  REQUIRE(data::Save("test_file.pgm", test) == true);

  // Now reload through our interface.
  REQUIRE(data::Load("test_file.pgm", test) == true);

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
TEST_CASE("LoadHDF5Test", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";
  arma::mat testTrans = trans(test);
  REQUIRE(testTrans.quiet_save("test_file.h5", arma::hdf5_binary)
      == true);
  REQUIRE(testTrans.quiet_save("test_file.hdf5", arma::hdf5_binary)
      == true);
  REQUIRE(testTrans.quiet_save("test_file.hdf", arma::hdf5_binary)
      == true);
  REQUIRE(testTrans.quiet_save("test_file.he5", arma::hdf5_binary)
      == true);

  // Now reload through our interface.
  REQUIRE(data::Load("test_file.h5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Make sure the other extensions work too.
  REQUIRE(data::Load("test_file.hdf5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  REQUIRE(data::Load("test_file.hdf", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  REQUIRE(data::Load("test_file.he5", test) == true);

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
TEST_CASE("SaveHDF5Test", "[LoadSaveTest]")
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";
  REQUIRE(data::Save("test_file.h5", test) == true);
  REQUIRE(data::Save("test_file.hdf5", test) == true);
  REQUIRE(data::Save("test_file.hdf", test) == true);
  REQUIRE(data::Save("test_file.he5", test) == true);

  // Now load them all and verify they were saved okay.
  REQUIRE(data::Load("test_file.h5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  // Make sure the other extensions work too.
  REQUIRE(data::Load("test_file.hdf5", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  REQUIRE(data::Load("test_file.hdf", test) == true);

  REQUIRE(test.n_rows == 4);
  REQUIRE(test.n_cols == 2);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(test[i] == Approx((double) (i + 1)).epsilon(1e-7));

  REQUIRE(data::Load("test_file.he5", test) == true);

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
TEST_CASE("NormalizeLabelSmallDatasetTest", "[LoadSaveTest]")
{
  arma::irowvec labels("-1 1 1 -1 -1 -1 1 1");
  arma::Row<size_t> newLabels;
  arma::ivec mappings;

  data::NormalizeLabels(labels, newLabels, mappings);

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

  data::RevertLabels(newLabels, mappings, revertedLabels);

  for (size_t i = 0; i < labels.n_elem; ++i)
    REQUIRE(labels[i] == revertedLabels[i]);
}

/**
 * Harder label normalization test.
 */
TEST_CASE("NormalizeLabelTest", "[LoadSaveTest]")
{
  arma::rowvec randLabels(5000);
  for (size_t i = 0; i < 5000; ++i)
    randLabels[i] = math::RandInt(-50, 50);
  randLabels[0] = 0.65; // Hey, doubles work too!

  arma::Row<size_t> newLabels;
  arma::vec mappings;

  data::NormalizeLabels(randLabels, newLabels, mappings);

  // Now map them back and ensure they are right.
  arma::rowvec revertedLabels(5000);
  data::RevertLabels(newLabels, mappings, revertedLabels);

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
    ar & BOOST_SERIALIZATION_NVP(c);
    ar & BOOST_SERIALIZATION_NVP(s);
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
    ar & BOOST_SERIALIZATION_NVP(x);
    ar & BOOST_SERIALIZATION_NVP(y);
    ar & BOOST_SERIALIZATION_NVP(ina);
    ar & BOOST_SERIALIZATION_NVP(inb);
  }

  // Public members for testing.
  int x;
  int y;
  TestInner ina;
  TestInner inb;
};

/**
 * Make sure we can load and save.
 */
TEST_CASE("LoadBinaryTest", "[LoadSaveTest]")
{
  Test x(10, 12);

  REQUIRE(data::Save("test.bin", "x", x, false) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(data::Load("test.bin", "x", y, false) == true);

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
TEST_CASE("LoadXMLTest", "[LoadSaveTest]")
{
  Test x(10, 12);

  REQUIRE(data::Save("test.xml", "x", x, false) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(data::Load("test.xml", "x", y, false) == true);

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
TEST_CASE("LoadTextTest", "[LoadSaveTest]")
{
  Test x(10, 12);

  REQUIRE(data::Save("test.txt", "x", x, false) == true);

  // Now reload.
  Test y(11, 14);

  REQUIRE(data::Load("test.txt", "x", y, false) == true);

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
TEST_CASE("DatasetInfoTest", "[LoadSaveTest]")
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
TEST_CASE("RegularCSVDatasetInfoLoad", "[LoadSaveTest]")
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
    DatasetInfo info;
    data::Load(testFiles[i], one);
    data::Load(testFiles[i], two, info);

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
      REQUIRE(info.Type(i) == Datatype::numeric);
  }
}

/**
 * Test non-transposed loading of regular CSVs with DatasetInfo.  Everything
 * should be numeric.
 */
TEST_CASE("NontransposedCSVDatasetInfoLoad", "[LoadSaveTest]")
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
    DatasetInfo info;
    data::Load(testFiles[i], one, true, false); // No transpose.
    data::Load(testFiles[i], two, info, true, false);

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
      REQUIRE(info.Type(i) == Datatype::numeric);
  }
}

/**
 * Create a file with a categorical string feature, then load it.
 */
TEST_CASE("CategoricalCSVLoadTest00", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.csv", matrix, info);

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

  REQUIRE(info.Type(0) == Datatype::numeric);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::categorical);

  REQUIRE(info.MapString<arma::uword>("hello", 2) == 0);
  REQUIRE(info.MapString<arma::uword>("goodbye", 2) == 1);
  REQUIRE(info.MapString<arma::uword>("coffee", 2) == 2);
  REQUIRE(info.MapString<arma::uword>("confusion", 2) == 3);

  REQUIRE(info.UnmapString(0, 2) == "hello");
  REQUIRE(info.UnmapString(1, 2) == "goodbye");
  REQUIRE(info.UnmapString(2, 2) == "coffee");
  REQUIRE(info.UnmapString(3, 2) == "confusion");

  remove("test.csv");
}

TEST_CASE("CategoricalCSVLoadTest01", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.csv", matrix, info, true);

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

  REQUIRE(info.Type(0) == Datatype::categorical);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::numeric);
  REQUIRE(info.Type(3) == Datatype::numeric);

  REQUIRE(info.MapString<arma::uword>("1", 0) == 0);
  REQUIRE(info.MapString<arma::uword>("", 0) == 1);

  REQUIRE(info.UnmapString(0, 0) == "1");
  REQUIRE(info.UnmapString(1, 0) == "");

  remove("test.csv");
}

TEST_CASE("CategoricalCSVLoadTest02", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.csv", matrix, info, true);

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

  REQUIRE(info.Type(0) == Datatype::categorical);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::numeric);

  REQUIRE(info.MapString<arma::uword>("", 0) == 1);
  REQUIRE(info.MapString<arma::uword>("1", 0) == 0);

  REQUIRE(info.UnmapString(0, 0) == "1");
  REQUIRE(info.UnmapString(1, 0) == "");

  remove("test.csv");
}

TEST_CASE("CategoricalCSVLoadTest03", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.csv", matrix, info, true);

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

  REQUIRE(info.Type(0) == Datatype::categorical);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::numeric);

  REQUIRE(info.MapString<arma::uword>("", 0) == 0);
  REQUIRE(info.MapString<arma::uword>("1", 0) == 1);

  REQUIRE(info.UnmapString(0, 0) == "");
  REQUIRE(info.UnmapString(1, 0) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalCSVLoadTest04", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.csv", matrix, info, true);

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

  REQUIRE(info.Type(0) == Datatype::categorical);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::numeric);

  REQUIRE(info.MapString<arma::uword>("200-DM", 0) == 0);
  REQUIRE(info.MapString<arma::uword>("1", 0) == 1);

  REQUIRE(info.UnmapString(0, 0) == "200-DM");
  REQUIRE(info.UnmapString(1, 0) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest00", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.csv", matrix, info, true, false); // No transpose.

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

  REQUIRE(info.Type(0) == Datatype::categorical);
  REQUIRE(info.Type(1) == Datatype::categorical);
  REQUIRE(info.Type(2) == Datatype::categorical);
  REQUIRE(info.Type(3) == Datatype::categorical);
  REQUIRE(info.Type(4) == Datatype::categorical);
  REQUIRE(info.Type(5) == Datatype::numeric);
  REQUIRE(info.Type(6) == Datatype::categorical);

  REQUIRE(info.MapString<arma::uword>("1", 0) == 0);
  REQUIRE(info.MapString<arma::uword>("2", 0) == 1);
  REQUIRE(info.MapString<arma::uword>("hello", 0) == 2);
  REQUIRE(info.MapString<arma::uword>("3", 1) == 0);
  REQUIRE(info.MapString<arma::uword>("4", 1) == 1);
  REQUIRE(info.MapString<arma::uword>("goodbye", 1) == 2);
  REQUIRE(info.MapString<arma::uword>("5", 2) == 0);
  REQUIRE(info.MapString<arma::uword>("6", 2) == 1);
  REQUIRE(info.MapString<arma::uword>("coffee", 2) == 2);
  REQUIRE(info.MapString<arma::uword>("7", 3) == 0);
  REQUIRE(info.MapString<arma::uword>("8", 3) == 1);
  REQUIRE(info.MapString<arma::uword>("confusion", 3) == 2);
  REQUIRE(info.MapString<arma::uword>("9", 4) == 0);
  REQUIRE(info.MapString<arma::uword>("10", 4) == 1);
  REQUIRE(info.MapString<arma::uword>("hello", 4) == 2);
  REQUIRE(info.MapString<arma::uword>("13", 6) == 0);
  REQUIRE(info.MapString<arma::uword>("14", 6) == 1);
  REQUIRE(info.MapString<arma::uword>("confusion", 6) == 2);

  REQUIRE(info.UnmapString(0, 0) == "1");
  REQUIRE(info.UnmapString(1, 0) == "2");
  REQUIRE(info.UnmapString(2, 0) == "hello");
  REQUIRE(info.UnmapString(0, 1) == "3");
  REQUIRE(info.UnmapString(1, 1) == "4");
  REQUIRE(info.UnmapString(2, 1) == "goodbye");
  REQUIRE(info.UnmapString(0, 2) == "5");
  REQUIRE(info.UnmapString(1, 2) == "6");
  REQUIRE(info.UnmapString(2, 2) == "coffee");
  REQUIRE(info.UnmapString(0, 3) == "7");
  REQUIRE(info.UnmapString(1, 3) == "8");
  REQUIRE(info.UnmapString(2, 3) == "confusion");
  REQUIRE(info.UnmapString(0, 4) == "9");
  REQUIRE(info.UnmapString(1, 4) == "10");
  REQUIRE(info.UnmapString(2, 4) == "hello");
  REQUIRE(info.UnmapString(0, 6) == "13");
  REQUIRE(info.UnmapString(1, 6) == "14");
  REQUIRE(info.UnmapString(2, 6) == "confusion");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest01", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.csv", matrix, info, true, false); // No transpose.

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

  REQUIRE(info.Type(0) == Datatype::numeric);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::categorical);
  REQUIRE(info.Type(3) == Datatype::numeric);

  REQUIRE(info.MapString<arma::uword>("", 2) == 0);
  REQUIRE(info.MapString<arma::uword>("1", 2) == 1);

  REQUIRE(info.UnmapString(0, 2) == "");
  REQUIRE(info.UnmapString(1, 2) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest02", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.csv", matrix, info, true, false); // No transpose.

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

  REQUIRE(info.Type(0) == Datatype::numeric);
  REQUIRE(info.Type(1) == Datatype::categorical);
  REQUIRE(info.Type(2) == Datatype::numeric);
  REQUIRE(info.Type(3) == Datatype::numeric);

  REQUIRE(info.MapString<arma::uword>("", 1) == 0);
  REQUIRE(info.MapString<arma::uword>("1", 1) == 1);

  REQUIRE(info.UnmapString(0, 1) == "");
  REQUIRE(info.UnmapString(1, 1) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest03", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.csv", matrix, info, true, false); // No transpose.

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

  REQUIRE(info.Type(0) == Datatype::categorical);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::numeric);
  REQUIRE(info.Type(3) == Datatype::numeric);

  REQUIRE(info.MapString<arma::uword>("", 1) == 0);
  REQUIRE(info.MapString<arma::uword>("1", 1) == 1);

  REQUIRE(info.UnmapString(0, 1) == "");
  REQUIRE(info.UnmapString(1, 1) == "1");

  remove("test.csv");
}

TEST_CASE("CategoricalNontransposedCSVLoadTest04", "[LoadSaveTest]")
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
    DatasetInfo info;
    data::Load("test.csv", matrix, info, true, false); // No transpose.

    REQUIRE(matrix.n_cols == 3);
    REQUIRE(matrix.n_rows == 4);

    REQUIRE(info.Type(0) == Datatype::categorical);
    REQUIRE(info.Type(1) == Datatype::numeric);
    REQUIRE(info.Type(2) == Datatype::numeric);
    REQUIRE(info.Type(3) == Datatype::numeric);

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

    REQUIRE(info.MapString<arma::uword>("200-DM", 1) == 0);
    REQUIRE(info.MapString<arma::uword>("1", 1) == 1);

    REQUIRE(info.UnmapString(0, 1) == "200-DM");
    REQUIRE(info.UnmapString(1, 1) == "1");

    remove("test.csv");
}

/**
 * A harder test CSV based on the concerns in #658.
 */
TEST_CASE("HarderKeonTest", "[LoadSaveTest]")
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
  data::DatasetInfo info;
  data::Load("test.csv", dataset, info, true, true);

  REQUIRE(dataset.n_rows == 5);
  REQUIRE(dataset.n_cols == 4);

  REQUIRE(info.Dimensionality() == 5);
  REQUIRE(info.NumMappings(0) == 3);
  REQUIRE(info.NumMappings(1) == 4);
  REQUIRE(info.NumMappings(2) == 0);
  REQUIRE(info.NumMappings(3) == 2); // \t and "" are equivalent.
  REQUIRE(info.NumMappings(4) == 4);

  // Now load non-transposed.
  data::DatasetInfo ntInfo;
  data::Load("test.csv", dataset, ntInfo, true, false);

  REQUIRE(dataset.n_rows == 4);
  REQUIRE(dataset.n_cols == 5);

  REQUIRE(ntInfo.Dimensionality() == 4);
  REQUIRE(ntInfo.NumMappings(0) == 4);
  REQUIRE(ntInfo.NumMappings(1) == 5);
  REQUIRE(ntInfo.NumMappings(2) == 5);
  REQUIRE(ntInfo.NumMappings(3) == 3);

  remove("test.csv");
}

/**
 * A simple ARFF load test.  Two attributes, both numeric.
 */
TEST_CASE("SimpleARFFTest", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.arff", dataset, info);

  REQUIRE(info.Dimensionality() == 2);
  REQUIRE(info.Type(0) == Datatype::numeric);
  REQUIRE(info.Type(1) == Datatype::numeric);

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
TEST_CASE("SimpleARFFCategoricalTest", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.arff", dataset, info);

  REQUIRE(info.Dimensionality() == 3);

  REQUIRE(info.Type(0) == Datatype::categorical);
  REQUIRE(info.NumMappings(0) == 3);
  REQUIRE(info.Type(1) == Datatype::numeric);
  REQUIRE(info.Type(2) == Datatype::categorical);
  REQUIRE(info.NumMappings(2) == 2);

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
TEST_CASE("HarderARFFTest", "[LoadSaveTest]")
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
  DatasetInfo info;
  data::Load("test.arff", dataset, info);

  REQUIRE(info.Dimensionality() == 5);

  REQUIRE(info.Type(0) == Datatype::numeric);

  REQUIRE(info.Type(1) == Datatype::categorical);
  REQUIRE(info.NumMappings(1) == 3);

  REQUIRE(info.Type(2) == Datatype::numeric);
  REQUIRE(info.Type(3) == Datatype::numeric);
  REQUIRE(info.Type(4) == Datatype::numeric);

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
TEST_CASE("BadDatasetInfoARFFTest", "[LoadSaveTest]")
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

  REQUIRE_THROWS_AS(data::LoadARFF("test.arff", dataset, info),
      std::invalid_argument);

  remove("test.arff");
}

/**
 * If file is not found, it should throw.
 */
TEST_CASE("NonExistentFileARFFTest", "[LoadSaveTest]")
{
  arma::mat dataset;
  DatasetInfo info;

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(data::LoadARFF("nonexistentfile.arff", dataset, info),
      std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * A test to check whether the arff loader is case insensitive to declarations:
 * @relation, @attribute, @data.
 */
TEST_CASE("CaseTest", "[LoadSaveTest]")
{
  arma::mat dataset;

  DatasetMapper<IncrementPolicy> info;

  LoadARFF<double, IncrementPolicy>("casecheck.arff", dataset, info);

  REQUIRE(dataset.n_rows == 2);
  REQUIRE(dataset.n_cols == 3);
}

/**
 * Ensure that a failure happens if we set a category to use capital letters but
 * it receives them in lowercase.
 */
TEST_CASE("CategoryCaseTest", "[LoadSaveTest]")
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
  data::DatasetInfo info;

  // Make sure to parse with fatal errors (that's what the `true` parameter
  // means).
  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(data::Load("test.arff", dataset, info, true),
      std::runtime_error);
  Log::Fatal.ignoreInput = false;

  remove("test.arff");
}

/**
 * Test that a CSV with the wrong number of columns fails.
 */
TEST_CASE("MalformedCSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7" << endl;
  f << "8, 9, 10, 11" << endl;
  f.close();

  arma::mat dataset;
  DatasetInfo di;

  REQUIRE(!data::Load("test.csv", dataset, di, false));

  remove("test.csv");
}

/**
 * Test that a TSV can load with LoadCSV.
 */
TEST_CASE("LoadCSVTSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test.tsv", fstream::out);
  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;
  f.close();

  arma::mat dataset;
  DatasetInfo di;

  REQUIRE(data::Load("test.tsv", dataset, di, false));

  REQUIRE(dataset.n_cols == 2);
  REQUIRE(dataset.n_rows == 4);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(dataset[i] == i + 1);

  remove("test.tsv");
}

/**
 * Test that a text file can load with LoadCSV.
 */
TEST_CASE("LoadCSVTXTTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test.txt", fstream::out);
  f << "1 2 3 4" << endl;
  f << "5 6 7 8" << endl;
  f.close();

  arma::mat dataset;
  DatasetInfo di;

  REQUIRE(data::Load("test.txt", dataset, di, false));

  REQUIRE(dataset.n_cols == 2);
  REQUIRE(dataset.n_rows == 4);

  for (size_t i = 0; i < 8; ++i)
    REQUIRE(dataset[i] == i + 1);

  remove("test.txt");
}

/**
 * Test that a non-transposed CSV with the wrong number of columns fails.
 */
TEST_CASE("MalformedNoTransposeCSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7" << endl;
  f << "8, 9, 10, 11" << endl;
  f.close();

  arma::mat dataset;
  DatasetInfo di;

  REQUIRE(!data::Load("test.csv", dataset, di, false, false));

  remove("test.csv");
}

/**
 * Test that a non-transposed TSV can load with LoadCSV.
 */
TEST_CASE("LoadCSVNoTransposeTSVTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test.tsv", fstream::out);
  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;
  f.close();

  arma::mat dataset;
  DatasetInfo di;

  REQUIRE(data::Load("test.tsv", dataset, di, false, false));

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
TEST_CASE("LoadCSVNoTransposeTXTTest", "[LoadSaveTest]")
{
  fstream f;
  f.open("test.txt", fstream::out);
  f << "1 2 3 4" << endl;
  f << "5 6 7 8" << endl;
  f.close();

  arma::mat dataset;
  DatasetInfo di;

  REQUIRE(data::Load("test.txt", dataset, di, false, false));

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
 * Make sure DatasetMapper properly unmaps from non-unique strings.
 */
TEST_CASE("DatasetMapperNonUniqueTest", "[LoadSaveTest]")
{
  DatasetMapper<MissingPolicy> dm(1);

  // Map a couple of strings; they'll map to quiet_NaN().
  dm.MapString<double>("0.5", 0); // No mapping created.
  dm.MapString<double>("hello", 0); // Mapping created.
  dm.MapString<double>("goodbye", 0);
  dm.MapString<double>("cheese", 0);

  double nan = std::numeric_limits<double>::quiet_NaN();
  REQUIRE(dm.NumMappings(0) == 3);
  REQUIRE(dm.NumUnmappings(nan, 0) == 3);

  REQUIRE(dm.UnmapString(nan, 0) == "hello");
  REQUIRE(dm.UnmapString(nan, 0, 0) == "hello");
  REQUIRE(dm.UnmapString(nan, 0, 1) == "goodbye");
  REQUIRE(dm.UnmapString(nan, 0, 2) == "cheese");
}
