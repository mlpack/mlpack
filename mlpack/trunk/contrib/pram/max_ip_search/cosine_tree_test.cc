#include <armadillo>
#include <string>
#include "general_spacetree.h"
#include "dconebound.h"
#include "gen_cosine_tree.h"

using namespace mlpack;
using namespace std;

PROGRAM_INFO("Cosine Tree Tester",
	     "This program tests the cosine tree construction",
	     "");


PARAM_STRING_REQ("r", "The data set to be indexed", "");
PARAM_FLAG("xx_print_tree", "The flag to print the tree", "");
PARAM_FLAG("some_flag", "some test flag", "");

int main (int argc, char *argv[]) {

  IO::ParseCommandLine(argc, argv);

  arma::mat rdata;
  string rfile = IO::GetParam<string>("r");

  IO::Info << "Loading files..." << endl;
  if (!data::Load(rfile.c_str(), rdata)) 
    IO::Fatal << "Data file " << rfile << " not found!" << endl;

  IO::Info << "Files loaded." << endl
	   << "Data (" << rdata.n_rows << ", "
	   << rdata.n_cols << ")" << endl;


  typedef GeneralBinarySpaceTree<DConeBound<>, arma::mat> CTreeType;

  arma::Col<size_t> old_from_new_data;

  CTreeType *test_tree
    = proximity::MakeGenCosineTree<CTreeType>(rdata, 20,
					      &old_from_new_data,
					      NULL);

  if (IO::HasParam("xx_print_tree")) {
    test_tree->Print();
  } else {
    IO::Info << "Tree built" << endl;
  }

  if (IO::HasParam("some_flag"))
    printf("The flag is working!\n");
}


