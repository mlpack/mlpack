/** @author Dongryeol Lee
 *
 *  @file bilinear_form.test.cc
 */

#undef BOOST_ALL_DYN_LINK
#include "fastlib/fastlib.h"
#include "boost/program_options.hpp"
#include "boost/test/included/unit_test.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/if.hpp"
#include "bilinear_form_estimator_dev.h"
#include "log_determinant_dev.h"

#ifdef EPETRA_MPI
#include "trilinos/Epetra_MpiComm.h"
#else
#include "trilinos/Epetra_SerialComm.h"
#endif

namespace fl {
namespace ml {
namespace bilinear_form_test {
class BilinearFormTestSuite : public boost::unit_test_framework::test_suite {

  public:

    class BilinearFormTest {
      public:

        BilinearFormTest() {
        }

        void RunTests() {

	  fprintf(stderr, "Running the tests:\n");

          // Call MPI Finalize.
          MPI_Finalize();
        }
    };

  public:

    BilinearFormTestSuite()
        : boost::unit_test_framework::test_suite("Bilinear form test suite") {

      // Create an instance of test.
      boost::shared_ptr<BilinearFormTest> instance(new BilinearFormTest());

      // Create the test cases.
      boost::unit_test_framework::test_case* bilinear_form_test_case
      = BOOST_CLASS_TEST_CASE(
          &BilinearFormTest::RunTests, instance);
      // add the test cases to the test suite
      add(bilinear_form_test_case);
    }
};
};
};
};

boost::unit_test_framework::test_suite*
init_unit_test_suite(int argc, char** argv) {

  // Initialize MPI.
#ifdef EPETRA_MPI
  MPI_Init(&argc, &argv);
#endif

  // create the top test suite
  boost::unit_test_framework::test_suite* top_test_suite
  = BOOST_TEST_SUITE("Bilinear form tests");

  if (argc != 2) {
    NOTIFY("Wrong number of arguments for tree test. Expected test input files directory. Returning NULL.");
    return NULL;
  }

  // add test suites to the top test suite
  std::string input_files_directory = argv[1];
  top_test_suite->add(new fl::ml::bilinear_form_test::BilinearFormTestSuite());
  return top_test_suite;
}
