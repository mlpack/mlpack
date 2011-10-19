#include <mlpack/core.h>
#include <mlpack/core/tree/spacetree.h>
#include <mlpack/core/tree/hrectbound.h>
#include <string>
#include <fstream>
#include <iostream>
#include <armadillo>

#include "kde_dual_tree.hpp"

PARAM_STRING_REQ ("reference_filename",
                  "CSV file containing the reference data set.",
                  "");

PARAM_INT_REQ ("leaf_maximum", "maximum number of leaves", "");

using namespace mlpack;

class Kernel
{
  Kernel () {}
  ~Kernel () {}
};

void recurse (tree::BinarySpaceTree<bound::HRectBound<2> >* node, size_t* parent)
{
  (*parent) += 1;
  if (node == NULL)
  {
    return;
  }
  std::cerr << "node " <<
                    *parent <<
                   " has " <<
                   node->count () <<
                   " points" <<
                   std::endl;
  recurse (node->left (), parent);
  recurse (node->right (), parent);
}

int main (int argc, char* argv[])
{
  CLI::ParseCommandLine (argc, argv);
  arma::mat data;
  //size_t leaf_max = CLI::GetParam<size_t> ("leaf_maximum");

  if (not data::Load (
            CLI::GetParam<std::string> ("reference_filename").c_str (),
            data))
  {
    Log::Fatal << "No reference file" << std::endl;
  }

  std::cerr << "we'll recurse now" << std::endl;
  tree::BinarySpaceTree<bound::HRectBound<2> > b = tree::BinarySpaceTree<bound::HRectBound<2> > (data);
  //size_t level = 0;

  /* perform a depth first search of the tree */
  //recurse (&b, &level);

  kde::KdeDualTree<Kernel> kde = kde::KdeDualTree<Kernel> (data);

  return EXIT_SUCCESS;
}
