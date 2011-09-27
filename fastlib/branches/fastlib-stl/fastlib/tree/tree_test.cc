/**
 * @file uselapack_test.cc
 *
 * Tests for LAPACK integration.
 */

#include "bounds.h"
#include "spacetree.h"
#include <stdlib.h>
#include <sys/time.h> //Random
#include "../../mlpack/core/kernels/lmetric.h"

#define BOOST_TEST_MODULE Tree_Test
#include <boost/test/unit_test.hpp>

#define MAX_POINTS 1000
#define DIMENSION_COUNT 2


using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::kernel;
using namespace mlpack::bound;

BOOST_AUTO_TEST_CASE(kd_tree_test) {
  //Generate the dataset.  Each n-dimensional point 
  timeval seed;
  gettimeofday(&seed, NULL);
  srand(seed.tv_sec);

  size_t size = rand()%MAX_POINTS;
  arma::mat dataset = arma::mat(DIMENSION_COUNT, size);

  for(size_t i = 0; i < size; i++)  //Up to 1000 random points
    for(size_t j = 0; j < DIMENSION_COUNT; j++)
      dataset(j, i) = (double)rand(); //Don't care about the range
  
  //Dataset created, now lets copy it.
  arma::mat datacopy = arma::mat(dataset); //We will need this copy to check
  std::vector<size_t> mapping;             //mappings
  std::vector<size_t> mappingOldToNew;
  
  //mapping = old index from new index
  BinarySpaceTree<DHrectBound<2> > root(dataset, mapping, mappingOldToNew);  
  BOOST_REQUIRE_EQUAL(root.count(), size);   


  //Now, check that the mappings are correct.
  for(size_t i = 0; i < size; i++) {
    size_t oldIndex = mapping[i];
    for(size_t j = 0; j < DIMENSION_COUNT; j++) 
      BOOST_REQUIRE_EQUAL(dataset(j,i), datacopy(j,oldIndex));
  }

  //Now check the reverse mappings...
  for(size_t i = 0; i < size; i++) {
    size_t newIndex = mappingOldToNew[i];
    for(size_t j = 0; j < DIMENSION_COUNT; j++) 
      BOOST_REQUIRE_EQUAL(dataset(j,newIndex), datacopy(j,i));
  }
  
}
