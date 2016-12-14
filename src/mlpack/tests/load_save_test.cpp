/**
 * @file load_save_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

BOOST_AUTO_TEST_SUITE(LoadSaveTest);

/**
 * Make sure failure occurs when no extension given.
 */
BOOST_AUTO_TEST_CASE(NoExtensionLoad)
{
  arma::mat out;
  Log::Warn.ignoreInput = true;
  BOOST_REQUIRE(data::Load("noextension", out) == false);
  Log::Warn.ignoreInput = false;
}

/**
 * Make sure failure occurs when no extension given.
 */
BOOST_AUTO_TEST_CASE(NoExtensionSave)
{
  arma::mat out;
  Log::Warn.ignoreInput = true;
  BOOST_REQUIRE(data::Save("noextension", out) == false);
  Log::Warn.ignoreInput = false;
}

/**
 * Make sure load fails if the file does not exist.
 */
BOOST_AUTO_TEST_CASE(NotExistLoad)
{
  arma::mat out;
  Log::Warn.ignoreInput = true;
  BOOST_REQUIRE(data::Load("nonexistentfile_______________.csv", out) == false);
  Log::Warn.ignoreInput = false;
}

/**
 * Make sure a CSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(LoadCSVTest)
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;

  f.close();

  arma::mat test;
  BOOST_REQUIRE(data::Load("test_file.csv", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure a TSV is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(LoadTSVTest)
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  BOOST_REQUIRE(data::Load("test_file.csv", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Test TSV loading with .tsv extension.
 */
BOOST_AUTO_TEST_CASE(LoadTSVExtensionTest)
{
  fstream f;
  f.open("test_file.tsv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  BOOST_REQUIRE(data::Load("test_file.tsv", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.tsv");
}

/**
 * Make sure a CSV is saved correctly.
 */
BOOST_AUTO_TEST_CASE(SaveCSVTest)
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  BOOST_REQUIRE(data::Save("test_file.csv", test) == true);

  // Load it in and make sure it is the same.
  arma::mat test2;
  BOOST_REQUIRE(data::Load("test_file.csv", test2) == true);

  BOOST_REQUIRE_EQUAL(test2.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test2.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test2[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure CSVs can be loaded in transposed form.
 */
BOOST_AUTO_TEST_CASE(LoadTransposedCSVTest)
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;

  f.close();

  arma::mat test;
  BOOST_REQUIRE(data::Load("test_file.csv", test, false, true) == true);

  BOOST_REQUIRE_EQUAL(test.n_cols, 2);
  BOOST_REQUIRE_EQUAL(test.n_rows, 4);

  for (size_t i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure TSVs can be loaded in transposed form.
 */
BOOST_AUTO_TEST_CASE(LoadTransposedTSVTest)
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  BOOST_REQUIRE(data::Load("test_file.csv", test, false, true) == true);

  BOOST_REQUIRE_EQUAL(test.n_cols, 2);
  BOOST_REQUIRE_EQUAL(test.n_rows, 4);

  for (size_t i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Check TSV loading with .tsv extension.
 */
BOOST_AUTO_TEST_CASE(LoadTransposedTSVExtensionTest)
{
  fstream f;
  f.open("test_file.tsv", fstream::out);

  f << "1\t2\t3\t4" << endl;
  f << "5\t6\t7\t8" << endl;

  f.close();

  arma::mat test;
  BOOST_REQUIRE(data::Load("test_file.tsv", test, false, true) == true);

  BOOST_REQUIRE_EQUAL(test.n_cols, 2);
  BOOST_REQUIRE_EQUAL(test.n_rows, 4);

  for (size_t i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.tsv");
}

/**
 * Make sure CSVs can be loaded in non-transposed form.
 */
BOOST_AUTO_TEST_CASE(LoadNonTransposedCSVTest)
{
  fstream f;
  f.open("test_file.csv", fstream::out);

  f << "1, 3, 5, 7" << endl;
  f << "2, 4, 6, 8" << endl;

  f.close();

  arma::mat test;
  BOOST_REQUIRE(data::Load("test_file.csv", test, false, false) == true);

  BOOST_REQUIRE_EQUAL(test.n_cols, 4);
  BOOST_REQUIRE_EQUAL(test.n_rows, 2);

  for (size_t i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure CSVs can be saved in non-transposed form.
 */
BOOST_AUTO_TEST_CASE(SaveNonTransposedCSVTest)
{
  arma::mat test = "1 2;"
                   "3 4;"
                   "5 6;"
                   "7 8;";

  BOOST_REQUIRE(data::Save("test_file.csv", test, false, false) == true);

  // Load it in and make sure it is in the same.
  arma::mat test2;
  BOOST_REQUIRE(data::Load("test_file.csv", test2, false, false) == true);

  BOOST_REQUIRE_EQUAL(test2.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test2.n_cols, 2);

  for (size_t i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], test2[i], 1e-5);

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure arma_ascii is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(LoadArmaASCIITest)
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  BOOST_REQUIRE(testTrans.save("test_file.txt", arma::arma_ascii));

  BOOST_REQUIRE(data::Load("test_file.txt", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure a CSV is saved correctly.
 */
BOOST_AUTO_TEST_CASE(SaveArmaASCIITest)
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  BOOST_REQUIRE(data::Save("test_file.txt", test) == true);

  // Load it in and make sure it is the same.
  BOOST_REQUIRE(data::Load("test_file.txt", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure raw_ascii is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(LoadRawASCIITest)
{
  fstream f;
  f.open("test_file.txt", fstream::out);

  f << "1 2 3 4" << endl;
  f << "5 6 7 8" << endl;

  f.close();

  arma::mat test;
  BOOST_REQUIRE(data::Load("test_file.txt", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure CSV is loaded correctly as .txt.
 */
BOOST_AUTO_TEST_CASE(LoadCSVTxtTest)
{
  fstream f;
  f.open("test_file.txt", fstream::out);

  f << "1, 2, 3, 4" << endl;
  f << "5, 6, 7, 8" << endl;

  f.close();

  arma::mat test;
  BOOST_REQUIRE(data::Load("test_file.txt", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.txt");
}

/**
 * Make sure arma_binary is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(LoadArmaBinaryTest)
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  BOOST_REQUIRE(testTrans.quiet_save("test_file.bin", arma::arma_binary)
      == true);

  // Now reload through our interface.
  BOOST_REQUIRE(data::Load("test_file.bin", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.bin");
}

/**
 * Make sure arma_binary is saved correctly.
 */
BOOST_AUTO_TEST_CASE(SaveArmaBinaryTest)
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  BOOST_REQUIRE(data::Save("test_file.bin", test) == true);

  BOOST_REQUIRE(data::Load("test_file.bin", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.bin");
}

/**
 * Make sure raw_binary is loaded correctly.
 */
BOOST_AUTO_TEST_CASE(LoadRawBinaryTest)
{
  arma::mat test = "1 2;"
                   "3 4;"
                   "5 6;"
                   "7 8;";

  arma::mat testTrans = trans(test);
  BOOST_REQUIRE(testTrans.quiet_save("test_file.bin", arma::raw_binary)
      == true);

  // Now reload through our interface.
  Log::Warn.ignoreInput = true;
  BOOST_REQUIRE(data::Load("test_file.bin", test) == true);
  Log::Warn.ignoreInput = false;

  BOOST_REQUIRE_EQUAL(test.n_rows, 1);
  BOOST_REQUIRE_EQUAL(test.n_cols, 8);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.bin");
}

/**
 * Make sure load as PGM is successful.
 */
BOOST_AUTO_TEST_CASE(LoadPGMBinaryTest)
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  arma::mat testTrans = trans(test);
  BOOST_REQUIRE(testTrans.quiet_save("test_file.pgm", arma::pgm_binary)
      == true);

  // Now reload through our interface.
  BOOST_REQUIRE(data::Load("test_file.pgm", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.pgm");
}

/**
 * Make sure save as PGM is successful.
 */
BOOST_AUTO_TEST_CASE(SavePGMBinaryTest)
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";

  BOOST_REQUIRE(data::Save("test_file.pgm", test) == true);

  // Now reload through our interface.
  BOOST_REQUIRE(data::Load("test_file.pgm", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; i++)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Remove the file.
  remove("test_file.pgm");
}

// Don't perform any HDF5 tests on Armadillo 4.300-4.400 (inclusive).  A bug
// causes loading to fail.
#if ((ARMA_VERSION_MAJOR == 4) && \
        (ARMA_VERSION_MINOR < 300 || ARMA_VERSION_MINOR > 400)) || \
    (ARMA_VERSION_MAJOR >= 5)

#if defined(ARMA_USE_HDF5)
/**
 * Make sure load as HDF5 is successful.
 */
BOOST_AUTO_TEST_CASE(LoadHDF5Test)
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";
  arma::mat testTrans = trans(test);
  BOOST_REQUIRE(testTrans.quiet_save("test_file.h5", arma::hdf5_binary)
      == true);
  BOOST_REQUIRE(testTrans.quiet_save("test_file.hdf5", arma::hdf5_binary)
      == true);
  BOOST_REQUIRE(testTrans.quiet_save("test_file.hdf", arma::hdf5_binary)
      == true);
  BOOST_REQUIRE(testTrans.quiet_save("test_file.he5", arma::hdf5_binary)
      == true);

  // Now reload through our interface.
  BOOST_REQUIRE(data::Load("test_file.h5", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Make sure the other extensions work too.
  BOOST_REQUIRE(data::Load("test_file.hdf5", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  BOOST_REQUIRE(data::Load("test_file.hdf", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  BOOST_REQUIRE(data::Load("test_file.he5", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  remove("test_file.h5");
  remove("test_file.hdf");
  remove("test_file.hdf5");
  remove("test_file.he5");
}

/**
 * Make sure save as HDF5 is successful.
 */
BOOST_AUTO_TEST_CASE(SaveHDF5Test)
{
  arma::mat test = "1 5;"
                   "2 6;"
                   "3 7;"
                   "4 8;";
  BOOST_REQUIRE(data::Save("test_file.h5", test) == true);
  BOOST_REQUIRE(data::Save("test_file.hdf5", test) == true);
  BOOST_REQUIRE(data::Save("test_file.hdf", test) == true);
  BOOST_REQUIRE(data::Save("test_file.he5", test) == true);

  // Now load them all and verify they were saved okay.
  BOOST_REQUIRE(data::Load("test_file.h5", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  // Make sure the other extensions work too.
  BOOST_REQUIRE(data::Load("test_file.hdf5", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  BOOST_REQUIRE(data::Load("test_file.hdf", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  BOOST_REQUIRE(data::Load("test_file.he5", test) == true);

  BOOST_REQUIRE_EQUAL(test.n_rows, 4);
  BOOST_REQUIRE_EQUAL(test.n_cols, 2);

  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(test[i], (double) (i + 1), 1e-5);

  remove("test_file.h5");
  remove("test_file.hdf");
  remove("test_file.hdf5");
  remove("test_file.he5");
}
#else
/**
 * Ensure saving as HDF5 fails.
 */
BOOST_AUTO_TEST_CASE(NoHDF5Test)
{
  arma::mat test;
  test.randu(5, 5);

  // Stop warnings.
  Log::Warn.ignoreInput = true;
  BOOST_REQUIRE(data::Save("test_file.h5", test) == false);
  BOOST_REQUIRE(data::Save("test_file.hdf5", test) == false);
  BOOST_REQUIRE(data::Save("test_file.hdf", test) == false);
  BOOST_REQUIRE(data::Save("test_file.he5", test) == false);
  Log::Warn.ignoreInput = false;
}
#endif

#endif

/**
 * Test normalization of labels.
 */
BOOST_AUTO_TEST_CASE(NormalizeLabelSmallDatasetTest)
{
  arma::irowvec labels("-1 1 1 -1 -1 -1 1 1");
  arma::Row<size_t> newLabels;
  arma::ivec mappings;

  data::NormalizeLabels(labels, newLabels, mappings);

  BOOST_REQUIRE_EQUAL(mappings[0], -1);
  BOOST_REQUIRE_EQUAL(mappings[1], 1);

  BOOST_REQUIRE_EQUAL(newLabels[0], 0);
  BOOST_REQUIRE_EQUAL(newLabels[1], 1);
  BOOST_REQUIRE_EQUAL(newLabels[2], 1);
  BOOST_REQUIRE_EQUAL(newLabels[3], 0);
  BOOST_REQUIRE_EQUAL(newLabels[4], 0);
  BOOST_REQUIRE_EQUAL(newLabels[5], 0);
  BOOST_REQUIRE_EQUAL(newLabels[6], 1);
  BOOST_REQUIRE_EQUAL(newLabels[7], 1);

  arma::irowvec revertedLabels;

  data::RevertLabels(newLabels, mappings, revertedLabels);

  for (size_t i = 0; i < labels.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(labels[i], revertedLabels[i]);
}

/**
 * Harder label normalization test.
 */
BOOST_AUTO_TEST_CASE(NormalizeLabelTest)
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
    BOOST_REQUIRE_EQUAL(randLabels[i], revertedLabels[i]);
}

// Test structures.
class TestInner
{
 public:
  TestInner(char c, string s) : c(c), s(s) { }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(c, "char");
    ar & data::CreateNVP(s, "string");
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
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(x, "x");
    ar & data::CreateNVP(y, "y");
    ar & data::CreateNVP(ina, "ina");
    ar & data::CreateNVP(inb, "inb");
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
BOOST_AUTO_TEST_CASE(LoadBinaryTest)
{
  Test x(10, 12);

  BOOST_REQUIRE_EQUAL(data::Save("test.bin", "x", x, false), true);

  // Now reload.
  Test y(11, 14);

  BOOST_REQUIRE_EQUAL(data::Load("test.bin", "x", y, false), true);

  BOOST_REQUIRE_EQUAL(y.x, x.x);
  BOOST_REQUIRE_EQUAL(y.y, x.y);
  BOOST_REQUIRE_EQUAL(y.ina.c, x.ina.c);
  BOOST_REQUIRE_EQUAL(y.ina.s, x.ina.s);
  BOOST_REQUIRE_EQUAL(y.inb.c, x.inb.c);
  BOOST_REQUIRE_EQUAL(y.inb.s, x.inb.s);
}

/**
 * Make sure we can load and save.
 */
BOOST_AUTO_TEST_CASE(LoadXMLTest)
{
  Test x(10, 12);

  BOOST_REQUIRE_EQUAL(data::Save("test.xml", "x", x, false), true);

  // Now reload.
  Test y(11, 14);

  BOOST_REQUIRE_EQUAL(data::Load("test.xml", "x", y, false), true);

  BOOST_REQUIRE_EQUAL(y.x, x.x);
  BOOST_REQUIRE_EQUAL(y.y, x.y);
  BOOST_REQUIRE_EQUAL(y.ina.c, x.ina.c);
  BOOST_REQUIRE_EQUAL(y.ina.s, x.ina.s);
  BOOST_REQUIRE_EQUAL(y.inb.c, x.inb.c);
  BOOST_REQUIRE_EQUAL(y.inb.s, x.inb.s);
}

/**
 * Make sure we can load and save.
 */
BOOST_AUTO_TEST_CASE(LoadTextTest)
{
  Test x(10, 12);

  BOOST_REQUIRE_EQUAL(data::Save("test.txt", "x", x, false), true);

  // Now reload.
  Test y(11, 14);

  BOOST_REQUIRE_EQUAL(data::Load("test.txt", "x", y, false), true);

  BOOST_REQUIRE_EQUAL(y.x, x.x);
  BOOST_REQUIRE_EQUAL(y.y, x.y);
  BOOST_REQUIRE_EQUAL(y.ina.c, x.ina.c);
  BOOST_REQUIRE_EQUAL(y.ina.s, x.ina.s);
  BOOST_REQUIRE_EQUAL(y.inb.c, x.inb.c);
  BOOST_REQUIRE_EQUAL(y.inb.s, x.inb.s);
}

/**
 * Test DatasetInfo by making a map for a dimension.
 */
BOOST_AUTO_TEST_CASE(DatasetInfoTest)
{
  DatasetInfo di(100);

  // Do all types default to numeric?
  for (size_t i = 0; i < 100; ++i)
  {
    BOOST_REQUIRE(di.Type(i) == Datatype::numeric);
    BOOST_REQUIRE_EQUAL(di.NumMappings(i), 0);
  }

  // Okay.  Add some mappings for dimension 3.
  const size_t first = di.MapString("test_mapping_1", 3);
  const size_t second = di.MapString("test_mapping_2", 3);
  const size_t third = di.MapString("test_mapping_3", 3);

  BOOST_REQUIRE_EQUAL(first, 0);
  BOOST_REQUIRE_EQUAL(second, 1);
  BOOST_REQUIRE_EQUAL(third, 2);

  // Now dimension 3 should be categorical.
  for (size_t i = 0; i < 100; ++i)
  {
    if (i == 3)
    {
      BOOST_REQUIRE(di.Type(i) == Datatype::categorical);
      BOOST_REQUIRE_EQUAL(di.NumMappings(i), 3);
    }
    else
    {
      BOOST_REQUIRE(di.Type(i) == Datatype::numeric);
      BOOST_REQUIRE_EQUAL(di.NumMappings(i), 0);
    }
  }

  // Get the mappings back.
  const string& strFirst = di.UnmapString(first, 3);
  const string& strSecond = di.UnmapString(second, 3);
  const string& strThird = di.UnmapString(third, 3);

  BOOST_REQUIRE_EQUAL(strFirst, "test_mapping_1");
  BOOST_REQUIRE_EQUAL(strSecond, "test_mapping_2");
  BOOST_REQUIRE_EQUAL(strThird, "test_mapping_3");
}

/**
 * Test loading regular CSV with DatasetInfo.  Everything should be numeric.
 */
BOOST_AUTO_TEST_CASE(RegularCSVDatasetInfoLoad)
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
    BOOST_REQUIRE_EQUAL(one.n_elem, two.n_elem);
    BOOST_REQUIRE_EQUAL(one.n_rows, two.n_rows);
    BOOST_REQUIRE_EQUAL(one.n_cols, two.n_cols);
    for (size_t i = 0; i < one.n_elem; ++i)
    {
      if (std::abs(one[i]) < 1e-8)
        BOOST_REQUIRE_SMALL(two[i], 1e-8);
      else
        BOOST_REQUIRE_CLOSE(one[i], two[i], 1e-8);
    }

    // Check that all dimensions are numeric.
    for (size_t i = 0; i < two.n_rows; ++i)
      BOOST_REQUIRE(info.Type(i) == Datatype::numeric);
  }
}

/**
 * Test non-transposed loading of regular CSVs with DatasetInfo.  Everything
 * should be numeric.
 */
BOOST_AUTO_TEST_CASE(NontransposedCSVDatasetInfoLoad)
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
    BOOST_REQUIRE_EQUAL(one.n_elem, two.n_elem);
    BOOST_REQUIRE_EQUAL(one.n_rows, two.n_rows);
    BOOST_REQUIRE_EQUAL(one.n_cols, two.n_cols);
    for (size_t i = 0; i < one.n_elem; ++i)
    {
      if (std::abs(one[i]) < 1e-8)
        BOOST_REQUIRE_SMALL(two[i], 1e-8);
      else
        BOOST_REQUIRE_CLOSE(one[i], two[i], 1e-8);
    }

    // Check that all dimensions are numeric.
    for (size_t i = 0; i < two.n_rows; ++i)
      BOOST_REQUIRE(info.Type(i) == Datatype::numeric);
  }
}

/**
 * Create a file with a categorical string feature, then load it.
 */
BOOST_AUTO_TEST_CASE(CategoricalCSVLoadTest00)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, 7);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 3);

  BOOST_REQUIRE_EQUAL(matrix(0, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 0), 2);
  BOOST_REQUIRE_EQUAL(matrix(2, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(0, 1), 3);
  BOOST_REQUIRE_EQUAL(matrix(1, 1), 4);
  BOOST_REQUIRE_EQUAL(matrix(2, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 2), 5);
  BOOST_REQUIRE_EQUAL(matrix(1, 2), 6);
  BOOST_REQUIRE_EQUAL(matrix(2, 2), 2);
  BOOST_REQUIRE_EQUAL(matrix(0, 3), 7);
  BOOST_REQUIRE_EQUAL(matrix(1, 3), 8);
  BOOST_REQUIRE_EQUAL(matrix(2, 3), 3);
  BOOST_REQUIRE_EQUAL(matrix(0, 4), 9);
  BOOST_REQUIRE_EQUAL(matrix(1, 4), 10);
  BOOST_REQUIRE_EQUAL(matrix(2, 4), 0);
  BOOST_REQUIRE_EQUAL(matrix(0, 5), 11);
  BOOST_REQUIRE_EQUAL(matrix(1, 5), 12);
  BOOST_REQUIRE_EQUAL(matrix(2, 5), 3);
  BOOST_REQUIRE_EQUAL(matrix(0, 6), 13);
  BOOST_REQUIRE_EQUAL(matrix(1, 6), 14);
  BOOST_REQUIRE_EQUAL(matrix(2, 6), 3);

  BOOST_REQUIRE(info.Type(0) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(1) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(2) == Datatype::categorical);

  BOOST_REQUIRE_EQUAL(info.MapString("hello", 2), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("goodbye", 2), 1);
  BOOST_REQUIRE_EQUAL(info.MapString("coffee", 2), 2);
  BOOST_REQUIRE_EQUAL(info.MapString("confusion", 2), 3);

  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 2), "hello");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 2), "goodbye");
  BOOST_REQUIRE_EQUAL(info.UnmapString(2, 2), "coffee");
  BOOST_REQUIRE_EQUAL(info.UnmapString(3, 2), "confusion");

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(CategoricalCSVLoadTest01)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, 4);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 3);

  BOOST_REQUIRE_EQUAL(matrix(0, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(0, 1), 0);
  BOOST_REQUIRE_EQUAL(matrix(0, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 3), 0);
  BOOST_REQUIRE_EQUAL(matrix(1, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 3), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 3), 1);

  BOOST_REQUIRE(info.Type(0) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(1) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(2) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(3) == Datatype::numeric);

  BOOST_REQUIRE_EQUAL(info.MapString("1", 0), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("", 0), 1);

  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 0), "1");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 0), "");

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(CategoricalCSVLoadTest02)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, 4);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 3);

  BOOST_REQUIRE_EQUAL(matrix(0, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(0, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 2), 0);
  BOOST_REQUIRE_EQUAL(matrix(0, 3), 0);
  BOOST_REQUIRE_EQUAL(matrix(1, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 3), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 3), 1);

  BOOST_REQUIRE(info.Type(0) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(1) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(2) == Datatype::numeric);

  BOOST_REQUIRE_EQUAL(info.MapString("", 0), 1);
  BOOST_REQUIRE_EQUAL(info.MapString("1", 0), 0);

  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 0), "1");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 0), "");

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(CategoricalCSVLoadTest03)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, 4);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 3);

  BOOST_REQUIRE_EQUAL(matrix(0, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(0, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 3), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 3), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 3), 1);

  BOOST_REQUIRE(info.Type(0) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(1) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(2) == Datatype::numeric);

  BOOST_REQUIRE_EQUAL(info.MapString("", 0), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("1", 0), 1);

  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 0), "");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 0), "1");

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(CategoricalNontransposedCSVLoadTest00)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, 3);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 7);

  BOOST_REQUIRE_EQUAL(matrix(0, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(0, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 2), 2);
  BOOST_REQUIRE_EQUAL(matrix(1, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(1, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 2), 2);
  BOOST_REQUIRE_EQUAL(matrix(2, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(2, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 2), 2);
  BOOST_REQUIRE_EQUAL(matrix(3, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(3, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 2), 2);
  BOOST_REQUIRE_EQUAL(matrix(4, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(4, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(4, 2), 2);
  BOOST_REQUIRE_EQUAL(matrix(5, 0), 11);
  BOOST_REQUIRE_EQUAL(matrix(5, 1), 12);
  BOOST_REQUIRE_EQUAL(matrix(5, 2), 15);
  BOOST_REQUIRE_EQUAL(matrix(6, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(6, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(6, 2), 2);

  BOOST_REQUIRE(info.Type(0) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(1) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(2) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(3) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(4) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(5) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(6) == Datatype::categorical);

  BOOST_REQUIRE_EQUAL(info.MapString("1", 0), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("2", 0), 1);
  BOOST_REQUIRE_EQUAL(info.MapString("hello", 0), 2);
  BOOST_REQUIRE_EQUAL(info.MapString("3", 1), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("4", 1), 1);
  BOOST_REQUIRE_EQUAL(info.MapString("goodbye", 1), 2);
  BOOST_REQUIRE_EQUAL(info.MapString("5", 2), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("6", 2), 1);
  BOOST_REQUIRE_EQUAL(info.MapString("coffee", 2), 2);
  BOOST_REQUIRE_EQUAL(info.MapString("7", 3), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("8", 3), 1);
  BOOST_REQUIRE_EQUAL(info.MapString("confusion", 3), 2);
  BOOST_REQUIRE_EQUAL(info.MapString("9", 4), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("10", 4), 1);
  BOOST_REQUIRE_EQUAL(info.MapString("hello", 4), 2);
  BOOST_REQUIRE_EQUAL(info.MapString("13", 6), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("14", 6), 1);
  BOOST_REQUIRE_EQUAL(info.MapString("confusion", 6), 2);

  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 0), "1");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 0), "2");
  BOOST_REQUIRE_EQUAL(info.UnmapString(2, 0), "hello");
  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 1), "3");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 1), "4");
  BOOST_REQUIRE_EQUAL(info.UnmapString(2, 1), "goodbye");
  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 2), "5");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 2), "6");
  BOOST_REQUIRE_EQUAL(info.UnmapString(2, 2), "coffee");
  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 3), "7");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 3), "8");
  BOOST_REQUIRE_EQUAL(info.UnmapString(2, 3), "confusion");
  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 4), "9");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 4), "10");
  BOOST_REQUIRE_EQUAL(info.UnmapString(2, 4), "hello");
  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 6), "13");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 6), "14");
  BOOST_REQUIRE_EQUAL(info.UnmapString(2, 6), "confusion");

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(CategoricalNontransposedCSVLoadTest01)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, 3);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 4);

  BOOST_REQUIRE_EQUAL(matrix(0, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(2, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 2), 1);

  BOOST_REQUIRE(info.Type(0) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(1) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(2) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(3) == Datatype::numeric);

  BOOST_REQUIRE_EQUAL(info.MapString("", 2), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("1", 2), 1);

  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 2), "");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 2), "1");

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(CategoricalNontransposedCSVLoadTest02)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, 3);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 4);

  BOOST_REQUIRE_EQUAL(matrix(0, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(1, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 2), 1);

  BOOST_REQUIRE(info.Type(0) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(1) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(2) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(3) == Datatype::numeric);

  BOOST_REQUIRE_EQUAL(info.MapString("", 1), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("1", 1), 1);

  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 1), "");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 1), "1");

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(CategoricalNontransposedCSVLoadTest03)
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

  BOOST_REQUIRE_EQUAL(matrix.n_cols, 3);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, 4);

  BOOST_REQUIRE_EQUAL(matrix(0, 0), 0);
  BOOST_REQUIRE_EQUAL(matrix(0, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(0, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(1, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(2, 2), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 0), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 1), 1);
  BOOST_REQUIRE_EQUAL(matrix(3, 2), 1);

  BOOST_REQUIRE(info.Type(0) == Datatype::categorical);
  BOOST_REQUIRE(info.Type(1) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(2) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(3) == Datatype::numeric);

  BOOST_REQUIRE_EQUAL(info.MapString("", 1), 0);
  BOOST_REQUIRE_EQUAL(info.MapString("1", 1), 1);

  BOOST_REQUIRE_EQUAL(info.UnmapString(0, 1), "");
  BOOST_REQUIRE_EQUAL(info.UnmapString(1, 1), "1");

  remove("test.csv");
}

/**
 * A harder test CSV based on the concerns in #658.
 */
BOOST_AUTO_TEST_CASE(HarderKeonTest)
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

  BOOST_REQUIRE_EQUAL(dataset.n_rows, 5);
  BOOST_REQUIRE_EQUAL(dataset.n_cols, 4);

  BOOST_REQUIRE_EQUAL(info.Dimensionality(), 5);
  BOOST_REQUIRE_EQUAL(info.NumMappings(0), 3);
  BOOST_REQUIRE_EQUAL(info.NumMappings(1), 4);
  BOOST_REQUIRE_EQUAL(info.NumMappings(2), 0);
  BOOST_REQUIRE_EQUAL(info.NumMappings(3), 2); // \t and "" are equivalent.
  BOOST_REQUIRE_EQUAL(info.NumMappings(4), 4);

  // Now load non-transposed.
  data::DatasetInfo ntInfo;
  data::Load("test.csv", dataset, ntInfo, true, false);

  BOOST_REQUIRE_EQUAL(dataset.n_rows, 4);
  BOOST_REQUIRE_EQUAL(dataset.n_cols, 5);

  BOOST_REQUIRE_EQUAL(ntInfo.Dimensionality(), 4);
  BOOST_REQUIRE_EQUAL(ntInfo.NumMappings(0), 4);
  BOOST_REQUIRE_EQUAL(ntInfo.NumMappings(1), 5);
  BOOST_REQUIRE_EQUAL(ntInfo.NumMappings(2), 5);
  BOOST_REQUIRE_EQUAL(ntInfo.NumMappings(3), 3);

  remove("test.csv");
}

/**
 * A simple ARFF load test.  Two attributes, both numeric.
 */
BOOST_AUTO_TEST_CASE(SimpleARFFTest)
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

  BOOST_REQUIRE_EQUAL(info.Dimensionality(), 2);
  BOOST_REQUIRE(info.Type(0) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(1) == Datatype::numeric);

  BOOST_REQUIRE_EQUAL(dataset.n_rows, 2);
  BOOST_REQUIRE_EQUAL(dataset.n_cols, 4);

  for (size_t i = 0; i < 8; ++i)
    BOOST_REQUIRE_CLOSE(dataset[i], double(i + 1), 1e-5);

  remove("test.arff");
}

/**
 * Another simple ARFF load test.  Three attributes, two categorical, one
 * numeric.
 */
BOOST_AUTO_TEST_CASE(SimpleARFFCategoricalTest)
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
  f << "\% a comment line " << endl;
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

  BOOST_REQUIRE_EQUAL(info.Dimensionality(), 3);

  BOOST_REQUIRE(info.Type(0) == Datatype::categorical);
  BOOST_REQUIRE_EQUAL(info.NumMappings(0), 3);
  BOOST_REQUIRE(info.Type(1) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(2) == Datatype::categorical);
  BOOST_REQUIRE_EQUAL(info.NumMappings(2), 2);

  BOOST_REQUIRE_EQUAL(dataset.n_rows, 3);
  BOOST_REQUIRE_EQUAL(dataset.n_cols, 4);

  // The first dimension must all be different (except the ones that are the
  // same).
  BOOST_REQUIRE_EQUAL(dataset(0, 0), dataset(0, 3));
  BOOST_REQUIRE_NE(dataset(0, 0), dataset(0, 1));
  BOOST_REQUIRE_NE(dataset(0, 1), dataset(0, 2));
  BOOST_REQUIRE_NE(dataset(0, 2), dataset(0, 0));

  BOOST_REQUIRE_CLOSE(dataset(1, 0), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(1, 1), 2.34, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(1, 2), 1.03e5, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(1, 3), -1.3, 1e-5);

  BOOST_REQUIRE_EQUAL(dataset(2, 0), dataset(2, 2));
  BOOST_REQUIRE_EQUAL(dataset(2, 1), dataset(2, 3));
  BOOST_REQUIRE_NE(dataset(2, 0), dataset(2, 1));

  remove("test.arff");
}

/**
 * A harder ARFF test, where we have each type of supported value, and some
 * random whitespace too.
 */
BOOST_AUTO_TEST_CASE(HarderARFFTest)
{
  fstream f;
  f.open("test.arff", fstream::out);
  f << "@relation    \t test" << endl;
  f << endl;
  f << endl;
  f << "@attribute @@@@flfl numeric" << endl;
  f << endl;
  f << "\% comment" << endl;
  f << "@attribute \"hello world\" string" << endl;
  f << "@attribute 12345 integer" << endl;
  f << "@attribute real real" << endl;
  f << "@attribute \"blah blah blah     \t \" numeric \% comment" << endl;
  f << "\% comment" << endl;
  f << "@data" << endl;
  f << "1, one, 3, 4.5, 6" << endl;
  f << "2, two, 4, 5.5, 7 \% comment" << endl;
  f << "3, \"three five, six\", 5, 6.5, 8" << endl;
  f.close();

  arma::mat dataset;
  DatasetInfo info;
  data::Load("test.arff", dataset, info);

  BOOST_REQUIRE_EQUAL(info.Dimensionality(), 5);

  BOOST_REQUIRE(info.Type(0) == Datatype::numeric);

  BOOST_REQUIRE(info.Type(1) == Datatype::categorical);
  BOOST_REQUIRE_EQUAL(info.NumMappings(1), 3);

  BOOST_REQUIRE(info.Type(2) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(3) == Datatype::numeric);
  BOOST_REQUIRE(info.Type(4) == Datatype::numeric);

  BOOST_REQUIRE_EQUAL(dataset.n_rows, 5);
  BOOST_REQUIRE_EQUAL(dataset.n_cols, 3);

  BOOST_REQUIRE_CLOSE(dataset(0, 0), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(0, 1), 2.0, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(0, 2), 3.0, 1e-5);

  BOOST_REQUIRE_NE(dataset(1, 0), dataset(1, 1));
  BOOST_REQUIRE_NE(dataset(1, 1), dataset(1, 2));
  BOOST_REQUIRE_NE(dataset(1, 0), dataset(1, 2));

  BOOST_REQUIRE_CLOSE(dataset(2, 0), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(2, 1), 4.0, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(2, 2), 5.0, 1e-5);

  BOOST_REQUIRE_CLOSE(dataset(3, 0), 4.5, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(3, 1), 5.5, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(3, 2), 6.5, 1e-5);

  BOOST_REQUIRE_CLOSE(dataset(4, 0), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(4, 1), 7.0, 1e-5);
  BOOST_REQUIRE_CLOSE(dataset(4, 2), 8.0, 1e-5);

  remove("test.arff");
}

/**
 * If we pass a bad DatasetInfo, it should throw.
 */
BOOST_AUTO_TEST_CASE(BadDatasetInfoARFFTest)
{
  fstream f;
  f.open("test.arff", fstream::out);
  f << "@relation    \t test" << endl;
  f << endl;
  f << endl;
  f << "@attribute @@@@flfl numeric" << endl;
  f << endl;
  f << "\% comment" << endl;
  f << "@attribute \"hello world\" string" << endl;
  f << "@attribute 12345 integer" << endl;
  f << "@attribute real real" << endl;
  f << "@attribute \"blah blah blah     \t \" numeric \% comment" << endl;
  f << "\% comment" << endl;
  f << "@data" << endl;
  f << "1, one, 3, 4.5, 6" << endl;
  f << "2, two, 4, 5.5, 7 \% comment" << endl;
  f << "3, \"three five, six\", 5, 6.5, 8" << endl;
  f.close();

  arma::mat dataset;
  DatasetInfo info(6);

  BOOST_REQUIRE_THROW(data::LoadARFF("test.arff", dataset, info),
      std::invalid_argument);

  remove("test.arff");
}

BOOST_AUTO_TEST_SUITE_END();
