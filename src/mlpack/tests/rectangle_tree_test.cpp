 
/**
 * @file tree_traits_test.cpp
 * @author Andrew Wells
 *
 * Tests for the RectangleTree class.  This should ensure that the class works correctly
 * and that subsequent changes don't break anything.  Because it's only used to test the trees,
 * it is slow.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/tree/tree_traits.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(RectangleTreeTest);

// Be careful!  When writing new tests, always get the boolean value and store
// it in a temporary, because the Boost unit test macros do weird things and
// will cause bizarre problems.

// Test the traits on RectangleTrees.
BOOST_AUTO_TEST_CASE(RectangeTreeTraitsTest)
{
  // Children may be overlapping.
  bool b = TreeTraits<RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
                      tree::RTreeDescentHeuristic,
                      NeighborSearchStat<NearestNeighborSort>,
                      arma::mat> >::HasOverlappingChildren;
  BOOST_REQUIRE_EQUAL(b, true);
  
  // Points are not contained in multiple levels.
  b = TreeTraits<RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
                      tree::RTreeDescentHeuristic,
                      NeighborSearchStat<NearestNeighborSort>,
                      arma::mat> >::HasSelfChildren;
  BOOST_REQUIRE_EQUAL(b, false);
}

BOOST_AUTO_TEST_CASE(RectangleTreeConstructionCountTest)
{
  arma::mat dataset;
  dataset.randu(3, 1000); // 1000 points in 3 dimensions.
  
  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
                      tree::RTreeDescentHeuristic,
                      NeighborSearchStat<NearestNeighborSort>,
                      arma::mat> tree(dataset, 20, 6, 5, 2, 0);
  BOOST_REQUIRE_EQUAL(tree.NumDescendants(), 1000);
}

std::vector<arma::vec*> getAllPointsInTree(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
                      tree::RTreeDescentHeuristic,
                      NeighborSearchStat<NearestNeighborSort>,
                      arma::mat>& tree)
{
  std::vector<arma::vec*> vec;
  if(tree.NumChildren() > 0) {
    for(size_t i = 0; i < tree.NumChildren(); i++) {
      std::vector<arma::vec*> tmp = getAllPointsInTree(*(tree.Children()[i]));
      vec.insert(vec.begin(), tmp.begin(), tmp.end());
    }
  } else {
    for(size_t i = 0; i < tree.Count(); i++) {
      arma::vec* c = new arma::vec(tree.Dataset().col(tree.Points()[i])); 
      vec.push_back(c);
    }
  }
  return vec;
}

BOOST_AUTO_TEST_CASE(RectangleTreeConstructionRepeatTest)
{
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.
  
  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
                      tree::RTreeDescentHeuristic,
                      NeighborSearchStat<NearestNeighborSort>,
                      arma::mat> tree(dataset, 20, 6, 5, 2, 0);

  std::vector<arma::vec*> allPoints = getAllPointsInTree(tree);
  for(size_t i = 0; i < allPoints.size(); i++) {
    for(size_t j = i+1; j < allPoints.size(); j++) {
      arma::vec v1 = *(allPoints[i]);
      arma::vec v2 = *(allPoints[j]);
      bool same = true;
      for(size_t k = 0; k < v1.n_rows; k++) {
	same &= (v1[k] == v2[k]);
      }
      assert(same != true);
    }
  }
  for(size_t i = 0; i < allPoints.size(); i++) {
    delete allPoints[i];
  }
}

bool checkContainment(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
                      tree::RTreeDescentHeuristic,
                      NeighborSearchStat<NearestNeighborSort>,
                      arma::mat>& tree)
{
  bool passed = true;
  if(tree.NumChildren() == 0) {
    for(size_t i = 0; i < tree.Count(); i++) {
      passed &= tree.Bound().Contains(tree.Dataset().unsafe_col(tree.Points()[i]));
    }
  } else {
    for(size_t i = 0; i < tree.NumChildren(); i++) {
      bool p1 = true;
      for(size_t j = 0; j < tree.Bound().Dim(); j++) {
	p1 &= tree.Bound()[j].Contains(tree.Children()[i]->Bound()[j]);
      }
      passed &= p1;
      passed &= checkContainment(*(tree.Children()[i]));
    }
  }  
  return passed;
}

BOOST_AUTO_TEST_CASE(RectangleTreeContainmentTest)
{
    arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.
  
  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
                      tree::RTreeDescentHeuristic,
                      NeighborSearchStat<NearestNeighborSort>,
                      arma::mat> tree(dataset, 20, 6, 5, 2, 0);
  assert(checkContainment(tree) == true);
}

BOOST_AUTO_TEST_CASE(SingleTreeTraverserTest)
{
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.
  arma::Mat<size_t> neighbors1;
  arma::mat distances1;
  arma::Mat<size_t> neighbors2;
  arma::mat distances2;
  
  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
                      tree::RTreeDescentHeuristic,
                      NeighborSearchStat<NearestNeighborSort>,
                      arma::mat> RTree(dataset, 20, 6, 5, 2, 0);

  // nearest neighbor search with the R tree.
  mlpack::neighbor::NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
        RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
	  	      tree::RTreeDescentHeuristic,
  		      NeighborSearchStat<NearestNeighborSort>,
  		      arma::mat> > allknn1(&RTree,
        dataset, true);
        
  allknn1.Search(5, neighbors1, distances1);

  // nearest neighbor search the naive way.
  mlpack::neighbor::AllkNN allknn2(dataset,
        true, true);

  allknn2.Search(5, neighbors2, distances2);
  
  for(size_t i = 0; i < neighbors1.size(); i++) {
    assert(neighbors1[i] == neighbors2[i]);
    assert(distances1[i] == distances2[i]);
  }
}


BOOST_AUTO_TEST_SUITE_END();
