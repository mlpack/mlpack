/**
 * @file load_save_test.cpp
 * @author Ryan Curtin
 *
 * Tests for data::Load() and data::Save().
 */
#include <sstream>

#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;

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
  std::fstream f;
  f.open("test_file.csv", std::fstream::out);

  f << "1, 2, 3, 4" << std::endl;
  f << "5, 6, 7, 8" << std::endl;

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
 * Make sure CSVs can be loaded in non-transposed form.
 */
BOOST_AUTO_TEST_CASE(LoadNonTransposedCSVTest)
{
  std::fstream f;
  f.open("test_file.csv", std::fstream::out);

  f << "1, 3, 5, 7" << std::endl;
  f << "2, 4, 6, 8" << std::endl;

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
  std::fstream f;
  f.open("test_file.txt", std::fstream::out);

  f << "1 2 3 4" << std::endl;
  f << "5 6 7 8" << std::endl;

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
  std::fstream f;
  f.open("test_file.txt", std::fstream::out);

  f << "1, 2, 3, 4" << std::endl;
  f << "5, 6, 7, 8" << std::endl;

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

#if defined(ARMA_USE_HDF5) && (ARMA_VERSION_MAJOR == 3 \
    || (ARMA_VERSION_MAJOR == 4 && (ARMA_VERSION_MINOR < 300 \
    ||  ARMA_VERSION_MINOR > 400)))
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

/**
 * Test normalization of labels.
 */
BOOST_AUTO_TEST_CASE(NormalizeLabelSmallDatasetTest)
{
  arma::ivec labels("-1 1 1 -1 -1 -1 1 1");
  arma::Col<size_t> newLabels;
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

  arma::ivec revertedLabels;

  data::RevertLabels(newLabels, mappings, revertedLabels);

  for (size_t i = 0; i < labels.n_elem; ++i)
    BOOST_REQUIRE_EQUAL(labels[i], revertedLabels[i]);
}

/**
 * Harder label normalization test.
 */
BOOST_AUTO_TEST_CASE(NormalizeLabelTest)
{
  arma::vec randLabels(5000);
  for (size_t i = 0; i < 5000; ++i)
    randLabels[i] = math::RandInt(-50, 50);
  randLabels[0] = 0.65; // Hey, doubles work too!

  arma::Col<size_t> newLabels;
  arma::vec mappings;

  data::NormalizeLabels(randLabels, newLabels, mappings);

  // Now map them back and ensure they are right.
  arma::vec revertedLabels(5000);
  data::RevertLabels(newLabels, mappings, revertedLabels);

  for (size_t i = 0; i < 5000; ++i)
    BOOST_REQUIRE_EQUAL(randLabels[i], revertedLabels[i]);
}

BOOST_AUTO_TEST_SUITE_END();
