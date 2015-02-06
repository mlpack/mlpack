/**
 * @file save_restore_model_test.cpp
 * @author Neil Slagle
 *
 * Here we have tests for the SaveRestoreModel class.
 */
#include <mlpack/core/util/save_restore_utility.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

#define ARGSTR(a) a,#a

using namespace mlpack;
using namespace mlpack::util;

BOOST_AUTO_TEST_SUITE(SaveRestoreUtilityTest);

/*
 * Exhibit proper save restore utility usage of child class proper usage.
 */
class SaveRestoreTest
{
 private:
  size_t anInt;
  SaveRestoreUtility saveRestore;

 public:
  SaveRestoreTest()
  {
    saveRestore = SaveRestoreUtility();
    anInt = 0;
  }

  bool SaveModel(std::string filename)
  {
    saveRestore.SaveParameter(anInt, "anInt");
    return saveRestore.WriteFile(filename);
  }

  bool LoadModel(std::string filename)
  {
    bool success = saveRestore.ReadFile(filename);
    if (success)
      anInt = saveRestore.LoadParameter(anInt, "anInt");

    return success;
  }

  size_t AnInt() { return anInt; }
  void AnInt(size_t s) { anInt = s; }
};

/**
 * Perform a save and restore on basic types.
 */
BOOST_AUTO_TEST_CASE(SaveBasicTypes)
{
  bool b = false;
  char c = 67;
  unsigned u = 34;
  size_t s = 12;
  short sh = 100;
  int i = -23;
  float f = -2.34f;
  double d = 3.14159;
  std::string cc = "Hello world!";

  SaveRestoreUtility sr;

  sr.SaveParameter(ARGSTR(b));
  sr.SaveParameter(ARGSTR(c));
  sr.SaveParameter(ARGSTR(u));
  sr.SaveParameter(ARGSTR(s));
  sr.SaveParameter(ARGSTR(sh));
  sr.SaveParameter(ARGSTR(i));
  sr.SaveParameter(ARGSTR(f));
  sr.SaveParameter(ARGSTR(d));
  sr.SaveParameter(ARGSTR(cc));
  sr.WriteFile("test_basic_types.xml");

  sr.ReadFile("test_basic_types.xml");

  bool b2 =         sr.LoadParameter(ARGSTR(b));
  char c2 =         sr.LoadParameter(ARGSTR(c));
  unsigned u2 =     sr.LoadParameter(ARGSTR(u));
  size_t s2 =       sr.LoadParameter(ARGSTR(s));
  short sh2 =       sr.LoadParameter(ARGSTR(sh));
  int i2 =          sr.LoadParameter(ARGSTR(i));
  float f2 =        sr.LoadParameter(ARGSTR(f));
  double d2 =       sr.LoadParameter(ARGSTR(d));
  std::string cc2 = sr.LoadParameter(ARGSTR(cc));

  BOOST_REQUIRE(b == b2);
  BOOST_REQUIRE(c == c2);
  BOOST_REQUIRE(u == u2);
  BOOST_REQUIRE(s == s2);
  BOOST_REQUIRE(sh == sh2);
  BOOST_REQUIRE(i == i2);
  BOOST_REQUIRE(cc == cc2);
  BOOST_REQUIRE_CLOSE(f, f2, 1e-5);
  BOOST_REQUIRE_CLOSE(d, d2, 1e-5);
}

BOOST_AUTO_TEST_CASE(SaveRestoreStdVector)
{
  size_t numbers[] = {0, 3, 6, 2, 6};
  std::vector<size_t> vec (numbers,
                           numbers + sizeof (numbers) / sizeof (size_t));
  SaveRestoreUtility sr;

  sr.SaveParameter(ARGSTR(vec));

  sr.WriteFile("test_std_vector_type.xml");

  sr.ReadFile("test_std_vector_type.xml");

  std::vector<size_t> loadee = sr.LoadParameter(ARGSTR(vec));

  for (size_t index = 0; index < loadee.size(); ++index)
    BOOST_REQUIRE_EQUAL(numbers[index], loadee[index]);
}

/**
 * Test the arma::mat functionality.
 */
BOOST_AUTO_TEST_CASE(SaveArmaMat)
{
  arma::mat matrix;
  matrix <<  1.2 << 2.3 << -0.1 << arma::endr
         <<  3.5 << 2.4 << -1.2 << arma::endr
         << -0.1 << 3.4 << -7.8 << arma::endr;

  SaveRestoreUtility sr;

  sr.SaveParameter(ARGSTR(matrix));

  sr.WriteFile("test_arma_mat_type.xml");

  sr.ReadFile("test_arma_mat_type.xml");

  arma::mat matrix2 = sr.LoadParameter(ARGSTR(matrix));

  for (size_t row = 0; row < matrix.n_rows; ++row)
    for (size_t column = 0; column < matrix.n_cols; ++column)
      BOOST_REQUIRE_CLOSE(matrix(row, column), matrix2(row, column), 1e-5);
}

/**
 * Test SaveRestoreModel proper usage in child classes and loading from
 *   separately defined objects
 */
BOOST_AUTO_TEST_CASE(SaveRestoreModelChildClassUsage)
{
  SaveRestoreTest saver;
  SaveRestoreTest loader;
  size_t s = 1200;
  const char* filename = "anInt.xml";

  saver.AnInt(s);
  saver.SaveModel(filename);

  loader.LoadModel(filename);

  BOOST_REQUIRE(loader.AnInt() == s);
}

BOOST_AUTO_TEST_SUITE_END();
