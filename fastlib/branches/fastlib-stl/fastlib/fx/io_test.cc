#include "optionshierarchy.h"
#include "io.h"
#include <string>
#include <iostream>
#include <sys/time.h>
#include <typeinfo>

#define DEFAULT_INT 42

#define BOOST_TEST_MODULE Something
#include <boost/test/unit_test.hpp>


using namespace mlpack;


/*
PROGRAM_INFO("MLPACK IO Test",
  "This is a simple test of the IO framework for input options and timers.  "
  "This particular text can be seen if you type --help.")

PARAM(int, "gint", "global desc", "global", 42, false); 
PARAM(int, "req", "required", "global", 23, true);
PARAM_INT("something_long_long_long", "A particularly long and needlessly "
  "verbose description ensures that my line hyphenator is working correctly "
  "but to test we need a "
  "really_really_really_long_long_long_long_long_long_long_long_word.", "",
  10);
*/

/**
* @brief Tests that inserting elements into an OptionsHierarchy
*   properly updates the tree.
*
* @return True indicating all is well with OptionsHierarchy
*/

namespace mlpack {
namespace io {

BOOST_AUTO_TEST_CASE(TestHierarchy) {
  OptionsHierarchy tmp = OptionsHierarchy("UTest");

  std::string testName = std::string("UTest/test");
  std::string testDesc = std::string("Test description.");
  std::string testTID = TYPENAME(int);

  //Check that the hierarchy is properly named.
  std::string str = std::string("UTest");
  OptionsData node = tmp.GetNodeData();
  
  BOOST_REQUIRE_EQUAL(str.compare(node.node), 0);

  //Check that inserting a node actually inserts the node.
  // Note, that since all versions of append simply call the most qualified
  //    overload, we will only test that one. 
  tmp.AppendNode(testName, testTID, testDesc);  
  BOOST_REQUIRE(tmp.FindNode(testName) != NULL);

  //Now check that the inserted node has the correct data.
  OptionsHierarchy* testHierarchy = tmp.FindNode(testName);
  OptionsData testData;
  if (testHierarchy != NULL) {
    node = testHierarchy->GetNodeData();
  
    BOOST_REQUIRE(testName.compare(node.node) == 0);
    BOOST_REQUIRE(testDesc.compare(node.desc) == 0);
    BOOST_REQUIRE(testTID.compare(node.tname) == 0);

  } else {}
}
}
}

/**
* @brief Tests that IO works as intended, namely that IO::Add
*   propogates successfully.
*
* @return True indicating all is well with IO, false otherwise.
*/
BOOST_AUTO_TEST_CASE(TestIO) {
  //  BOOST_REQUIRE_CLOSE(IO::GetParam<int>("global/gint") + 1e-6,
  //   DEFAULT_INT + 1e-6, 1e-5); 
  
  //Check that the IO::HasParam returns false if no value has been specified
  //On the commandline or programmatically.
  IO::Add<bool>("bool", "True or False", "global");

  BOOST_REQUIRE_EQUAL(IO::HasParam("global/bool"), false);


  IO::GetParam<bool>("global/bool") = true;

  //IO::HasParam should return true now.
  BOOST_REQUIRE_EQUAL(IO::HasParam("global/bool"), true);

  BOOST_REQUIRE_EQUAL(IO::GetDescription("global/bool").compare(
                              std::string("True or False")) , 0);

  //Check that SanitizeString is sanitary. 
  std::string tmp = IO::SanitizeString("/foo/bar/fizz");
  BOOST_REQUIRE_EQUAL(tmp.compare(std::string("foo/bar/fizz/")),0);

  IO::Add("global/tmp", "desc");
  std::vector<std::string> folder = IO::GetFolder("global");

//  IO::Info << folder.size() << std::endl;
//  for(std::vector<std::string>::iterator i = folder.begin();
//  i != folder.end(); i++)
//    IO::Info << *i << std::endl;
/*
  //Now lets test the output functions.  Will have to eyeball it manually.
  IO::Debug << "Test the new lines...";
  IO::Debug << "shouldn't get 'Info' here." << std::endl;
  IO::Debug << "But now I should." << std::endl << std::endl;

  //Test IO::Debug 
  IO::Debug << "You shouldn't see this when DEBUG=OFF" << std::endl;
*/

}

/**
* @brief Tests that the various PARAM_* macros work properly
* @return True indicating that all is well with IO & Options.
*/
BOOST_AUTO_TEST_CASE(TestOption) {
  //This test will involve creating an option, and making sure IO reflects this.
  PARAM(int, "test", "test desc", "test_parent", DEFAULT_INT, false);
  
  //Does IO reflect this?
  BOOST_REQUIRE_EQUAL(IO::HasParam("test_parent/test"), true);
   
  std::string desc = std::string("test desc");
  
  BOOST_REQUIRE(desc.compare(IO::GetDescription("test_parent/test"))==0);
  BOOST_REQUIRE(IO::GetParam<int>("test_parent/test") == DEFAULT_INT);
}


