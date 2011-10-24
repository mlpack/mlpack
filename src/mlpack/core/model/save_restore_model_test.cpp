/***
 * @file save_restore_model_test.cpp
 * @author Neil Slagle
 *
 * Here we have tests for the SaveRestoreModel class.
 */

#include "save_restore_model.hpp"

#define BOOST_TEST_MODULE SaveRestoreModel Test
#include <boost/test/unit_test.hpp>

#define ARGSTR(a) a,#a

/***
 * We must override the purely virtual method solve; the
 *  overridden saveModel and loadModel enable testing
 *  of child class proper usage
 */
class SaveRestoreModelTest : public SaveRestoreModel
{
  private:
    size_t anInt;
  public:
    bool solve () { return true; }
    bool saveModel (std::string filename)
    {
      this->saveParameter (anInt, "anInt");
      return this->writeFile (filename);
    }
    bool loadModel (std::string filename)
    {
      bool success = this->readFile (filename);
      if (success)
      {
        anInt = this->loadParameter (anInt, "anInt");
      }
      return success;
    }
    size_t getAnInt () { return anInt; }
    void setAnInt (size_t s) { this->anInt = s; }
};

/***
 * Perform a save and restore on basic types.
 */
BOOST_AUTO_TEST_CASE(save_basic_types)
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

  SaveRestoreModelTest* sRM = new SaveRestoreModelTest();

  sRM->saveParameter (ARGSTR(b));
  sRM->saveParameter (ARGSTR(c));
  sRM->saveParameter (ARGSTR(u));
  sRM->saveParameter (ARGSTR(s));
  sRM->saveParameter (ARGSTR(sh));
  sRM->saveParameter (ARGSTR(i));
  sRM->saveParameter (ARGSTR(f));
  sRM->saveParameter (ARGSTR(d));
  sRM->saveParameter (ARGSTR(cc));
  sRM->writeFile ("test_basic_types.xml");

  sRM->readFile ("test_basic_types.xml");

  bool b2 =         sRM->loadParameter (ARGSTR(b));
  char c2 =         sRM->loadParameter (ARGSTR(c));
  unsigned u2 =     sRM->loadParameter (ARGSTR(u));
  size_t s2 =       sRM->loadParameter (ARGSTR(s));
  short sh2 =       sRM->loadParameter (ARGSTR(sh));
  int i2 =          sRM->loadParameter (ARGSTR(i));
  float f2 =        sRM->loadParameter (ARGSTR(f));
  double d2 =       sRM->loadParameter (ARGSTR(d));
  std::string cc2 = sRM->loadParameter (ARGSTR(cc));

  BOOST_REQUIRE (b == b2);
  BOOST_REQUIRE (c == c2);
  BOOST_REQUIRE (u == u2);
  BOOST_REQUIRE (s == s2);
  BOOST_REQUIRE (sh == sh2);
  BOOST_REQUIRE (i == i2);
  BOOST_REQUIRE_CLOSE (f, f2, 1e-5);
  BOOST_REQUIRE_CLOSE (d, d2, 1e-5);

  delete sRM;
}

/***
 * Test the arma::mat functionality.
 */
BOOST_AUTO_TEST_CASE(save_arma_mat)
{
  arma::mat matrix;
  matrix <<  1.2 << 2.3 << -0.1 << arma::endr
         <<  3.5 << 2.4 << -1.2 << arma::endr
         << -0.1 << 3.4 << -7.8 << arma::endr;
  SaveRestoreModelTest* sRM = new SaveRestoreModelTest();

  sRM->saveParameter (ARGSTR (matrix));

  sRM->writeFile ("test_arma_mat_type.xml");

  sRM->readFile ("test_arma_mat_type.xml");

  arma::mat matrix2 = sRM->loadParameter (ARGSTR (matrix));

  for (size_t row = 0; row < matrix.n_rows; ++row)
  {
    for (size_t column = 0; column < matrix.n_cols; ++column)
    {
      BOOST_REQUIRE_CLOSE(matrix(row,column), matrix2(row,column), 1e-5);
    }
  }

  delete sRM;
}
/***
 * Test SaveRestoreModel proper usage in child classes and loading from
 *   separately defined objects
 */
BOOST_AUTO_TEST_CASE(save_restore_model_child_class_usage)
{
  SaveRestoreModelTest* saver = new SaveRestoreModelTest();
  SaveRestoreModelTest* loader = new SaveRestoreModelTest();
  size_t s = 1200;
  const char* filename = "anInt.xml";

  saver->setAnInt (s);
  saver->saveModel (filename);
  delete saver;

  loader->loadModel (filename);

  BOOST_REQUIRE (loader->getAnInt () == s);

  delete loader;
}
