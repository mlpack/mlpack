/**
 * @file save_restore_model_test.cpp
 * @author Neil Slagle
 *
 * Here we have tests for the SaveRestoreModel class.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core/util/save_restore_utility.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

#define ARGSTR(a) a,#a

using namespace mlpack;
using namespace mlpack::util;

BOOST_AUTO_TEST_SUITE(SaveRestoreUtilityTests);

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
  void AnInt(size_t s) { this->anInt = s; }
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

  SaveRestoreUtility* sRM = new SaveRestoreUtility();

  sRM->SaveParameter(ARGSTR(b));
  sRM->SaveParameter(ARGSTR(c));
  sRM->SaveParameter(ARGSTR(u));
  sRM->SaveParameter(ARGSTR(s));
  sRM->SaveParameter(ARGSTR(sh));
  sRM->SaveParameter(ARGSTR(i));
  sRM->SaveParameter(ARGSTR(f));
  sRM->SaveParameter(ARGSTR(d));
  sRM->SaveParameter(ARGSTR(cc));
  sRM->WriteFile("test_basic_types.xml");

  sRM->ReadFile("test_basic_types.xml");

  bool b2 =         sRM->LoadParameter(ARGSTR(b));
  char c2 =         sRM->LoadParameter(ARGSTR(c));
  unsigned u2 =     sRM->LoadParameter(ARGSTR(u));
  size_t s2 =       sRM->LoadParameter(ARGSTR(s));
  short sh2 =       sRM->LoadParameter(ARGSTR(sh));
  int i2 =          sRM->LoadParameter(ARGSTR(i));
  float f2 =        sRM->LoadParameter(ARGSTR(f));
  double d2 =       sRM->LoadParameter(ARGSTR(d));
  std::string cc2 = sRM->LoadParameter(ARGSTR(cc));

  BOOST_REQUIRE(b == b2);
  BOOST_REQUIRE(c == c2);
  BOOST_REQUIRE(u == u2);
  BOOST_REQUIRE(s == s2);
  BOOST_REQUIRE(sh == sh2);
  BOOST_REQUIRE(i == i2);
  BOOST_REQUIRE(cc == cc2);
  BOOST_REQUIRE_CLOSE(f, f2, 1e-5);
  BOOST_REQUIRE_CLOSE(d, d2, 1e-5);

  delete sRM;
}

BOOST_AUTO_TEST_CASE(SaveRestoreStdVector)
{
  size_t numbers[] = {0,3,6,2,6};
  std::vector<size_t> vec (numbers,
                           numbers + sizeof (numbers) / sizeof (size_t));
  SaveRestoreUtility* sRM = new SaveRestoreUtility();

  sRM->SaveParameter(ARGSTR(vec));

  sRM->WriteFile("test_std_vector_type.xml");

  sRM->ReadFile("test_std_vector_type.xml");

  std::vector<size_t> loadee = sRM->LoadParameter(ARGSTR(vec));

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

  SaveRestoreUtility* sRM = new SaveRestoreUtility();

  sRM->SaveParameter(ARGSTR(matrix));

  sRM->WriteFile("test_arma_mat_type.xml");

  sRM->ReadFile("test_arma_mat_type.xml");

  arma::mat matrix2 = sRM->LoadParameter(ARGSTR(matrix));

  for (size_t row = 0; row < matrix.n_rows; ++row)
    for (size_t column = 0; column < matrix.n_cols; ++column)
      BOOST_REQUIRE_CLOSE(matrix(row,column), matrix2(row,column), 1e-5);

  delete sRM;
}

/**
 * Test SaveRestoreModel proper usage in child classes and loading from
 *   separately defined objects
 */
BOOST_AUTO_TEST_CASE(SaveRestoreModelChildClassUsage)
{
  SaveRestoreTest* saver = new SaveRestoreTest();
  SaveRestoreTest* loader = new SaveRestoreTest();
  size_t s = 1200;
  const char* filename = "anInt.xml";

  saver->AnInt(s);
  saver->SaveModel(filename);
  delete saver;

  loader->LoadModel(filename);

  BOOST_REQUIRE(loader->AnInt() == s);

  delete loader;
}

BOOST_AUTO_TEST_SUITE_END();
