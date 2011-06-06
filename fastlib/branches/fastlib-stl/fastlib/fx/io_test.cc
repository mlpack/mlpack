#include "io.h"
#include "optionshierarchy.h"
#include "printing.h"

#include <string>
#include <iostream>
#include <sys/time.h>
#include <typeinfo>

#define DEFAULT_INT 42



namespace mlpack {
namespace io {

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

bool ASSERT(bool expression, const char* msg);
void TestAll();
bool TestHierarchy();
bool TestIO();
bool TestOption();


/**
* @brief Runs all the other tests, printing output as appropriate.
*/
void TestAll() {  
  IO::StartTimer("TestTimer");
  if (TestIO())
    IO::Info << "Test IO Succeeded." << std::endl;
  else
    IO::Fatal << "Test IO Failed." << std::endl;

  if (TestHierarchy())
    IO::Info << "Test Hierarchy Passed." << std::endl;
  else
    IO::Fatal << "Test Hierarchy Failed." << std::endl;

  if (TestOption())
    IO::Info << "Test Option Passed." << std::endl;
  else
    IO::Fatal << "Test Option Failed." << std::endl;
  IO::StopTimer("TestTimer");

  IO::Info << "Elapsed uSecs: " << IO::GetValue<timeval>("TestTimer").tv_usec 
            << std::endl;
}


/**
* @brief Tests that IO works as intended, namely that IO::Add
*   propogates successfully.
*
* @return True indicating all is well with IO, false otherwise.
*/
bool TestIO() {
  bool success = true;
  success = success & ASSERT(IO::GetValue<int>("global/gint") == DEFAULT_INT,
                    "IO::GetValue failed on gint");
   
  
  //Check that the IO::CheckValue returns false if no value has been specified
  //On the commandline or programmatically.
  IO::Add<bool>("bool", "True or False", "global");
  success = success & ASSERT(IO::CheckValue("global/bool") == false, 
                    "IO::CheckValue failed on global/bool");

  IO::GetValue<bool>("global/bool") = true;

  //IO::CheckValue should return true now.
  success = success & ASSERT(IO::CheckValue("global/bool") == true, 
                              "IO::CheckValue failed on global/bool #2");
  success = success & ASSERT(IO::GetValue<bool>("global/bool") == true, 
                    "IO::GetValue failed on global/bool");
  success = success & ASSERT(IO::GetDescription("global/bool").compare(
                              std::string("True or False")) == 0, 
                            "IO::GetDescription failed on global/bool");

  //Check that SanitizeString is sanitary. 
  std::string tmp = IO::SanitizeString("/foo/bar/fizz");
  success = success & ASSERT(tmp.compare(std::string("foo/bar/fizz/")) == 0, 
                              "IO::SanitizeString failed on 'foo/bar/fizz'");

  //Now lets test the output functions.  Will have to eyeball it manually.
  IO::Debug << "Test the new lines...";
  IO::Debug << "shouldn't get 'Info' here." << std::endl;
  IO::Debug << "But now I should." << std::endl << std::endl;

  //Test IO::Debug 
  IO::Debug << "You shouldn't see this when DEBUG=OFF" << std::endl;


  return success;
}


/**
* @brief Tests that inserting elements into an OptionsHierarchy
*   properly updates the tree.
*
* @return True indicating all is well with OptionsHierarchy
*/
bool TestHierarchy() {
  bool success = true;
  OptionsHierarchy tmp = OptionsHierarchy("UTest");
  std::string testName = std::string("UTest/test");
  std::string testDesc = std::string("Test description.");
  std::string testTID = TYPENAME(int);

  //Check that the hierarchy is properly named.
  std::string str = std::string("UTest");
  OptionsData node = tmp.GetNodeData();
  success = success & ASSERT(str.compare(node.node) == 0,
                              "OptionsHierarchy::GetNodeData failed on UTest");

  //Check that inserting a node actually inserts the node.
  /* Note, that since all versions of append simply call the most qualified
      overload, we will only test that one. */
  tmp.AppendNode(testName, testTID, testDesc);
  success = success & ASSERT(tmp.FindNode(testName) != NULL, 
                             "OptionsHierarchy::FindNode failed on UTest/test");

  //Now check that the inserted node has the correct data.
  OptionsHierarchy* testHierarchy = tmp.FindNode(testName);
  OptionsData testData;
  if (testHierarchy != NULL) {
    node = testHierarchy->GetNodeData();
    success = success & ASSERT(testName.compare(node.node) == 0 &&
                                testDesc.compare(node.desc) == 0 &&
                                testTID.compare(node.tname) == 0,
                          "OptionsHierarchy::GetNodeData failed on UTest/test");
  } else 
    success = false;
  return success;
}


/**
* @brief Tests that the various PARAM_* macros work properly.
*
* @return True indicating that all is well with IO & Options.
*/
bool TestOption() {
  
  bool success = true;
  //This test will involve creating an option, and making sure IO reflects this.
  PARAM(int, "test", "test desc", "test_parent", DEFAULT_INT, false);
  
  //Does IO reflect this?
  success = success & ASSERT(IO::CheckValue("test_parent/test"),
                              "IO::CheckValue failed on parent/test");
   

  std::string desc = std::string("test desc");
  success = success & 
    ASSERT(desc.compare(IO::GetDescription("test_parent/test")) == 0, 
            "IO::GetDescription fails on test_parent/test");

  success = success & 
    ASSERT(IO::GetValue<int>("test_parent/test") == DEFAULT_INT, 
            "IO::GetValue fails on test_parent/test");

  return success;
}


/**
* @brief If the expression is true, true is returned.  Otherwise
*   an error message is printed and failure is returned.
*
* @param expression The expression to be evaluated.
* @param msg The error message printed on failure.
*
* @return True indicating success and false indicates failure.
*/
bool ASSERT(bool expression, const char* msg) {
  if (!expression) {
    IO::Fatal << msg << std::endl; 
    return false;
  }
  return true;
}

}; //namespace io
}; //namespace mlpack

int main(int argc, char** argv) {
  mlpack::io::TestAll();
  mlpack::IO::ParseCommandLine(argc, argv);
  mlpack::IO::Warn << "Application did not terminate..." << std::endl;
}
