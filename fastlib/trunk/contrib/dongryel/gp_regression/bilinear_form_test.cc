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

        void RandomDataset(GenMatrix<double, false> *random_dataset) {
	}

        void RunTests() {

	  fprintf(stderr, "Running the tests:\n");

	  // Generate a random table.
	  GenMatrix<double, false> random_dataset;
	  RandomDataset(&random_dataset);
	  
	  // Generate a random kernel.
	  //	  typedef fl::math::GaussianDotProduct< double, fl::math::LMetric<2> > KernelType;
	  //fl::math::LMetric<2> metric;
	  KernelType kernel;
	  kernel.Init(math::Random<double>(1, 10), &metric);
	  printf("Testing on the Gaussian kernel with the bandwidth of %g.\n",
		 kernel.bandwidth() );
	  
	  // Make a kernel matrix linear operator.
#ifdef EPETRA_MPI
	  Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
	  Epetra_SerialComm comm;
#endif
	  Epetra_Map map(random_dataset.n_entries(), 0, comm);
	  Anasazi::KernelLinearOperator<KernelType, false, false> op(
	     random_dataset, kernel, comm, map);
	  
	  // Make a Lanczos object, and run it.
	  fl::ml::BilinearFormEstimator<fl::ml::InverseTransformation> bilinear;
	  bilinear.Init(&op);
	  
	  // A random intitial starting vector, and with it compute the
	  // Lanczos tridiagonal matrix.
	  Vector random_initial_vector;
	  RandomVector_(random_dataset.n_entries(), &random_initial_vector);
	  
	  // Test the log determinant computation.
	  fl::ml::LogDeterminant log_determinant;
	  log_determinant.Init(&op);
	  log_determinant.set_max_num_iterations(3);
	  
	  printf("Testing the log determinant: \n");
	  printf("-----------------------------\n");
	  printf("The ultra naive estimate should be %g.\n",
		 log_determinant.NaiveCompute() );
	  printf("The naive estimate is %g.\n",
		 log_determinant.Compute() );
	  printf("The Monte Carlo estimate is %g.\n", 
		 log_determinant.MonteCarloCompute() );

          // Call MPI Finalize.
#ifdef EPETRA_MPI
          MPI_Finalize();
#endif
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
