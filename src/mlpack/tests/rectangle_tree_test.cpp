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

BOOST_AUTO_TEST_CASE(RectangeTreeTraitsTest) {
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

// Test to make sure the tree can be contains the correct number of points after it is
// constructed.

BOOST_AUTO_TEST_CASE(RectangleTreeConstructionCountTest) {
  arma::mat dataset;
  dataset.randu(3, 1000); // 1000 points in 3 dimensions.

  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> tree(dataset, 20, 6, 5, 2, 0);
    RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> tree2 = tree;
	  
  BOOST_REQUIRE_EQUAL(tree.NumDescendants(), 1000);
  BOOST_REQUIRE_EQUAL(tree2.NumDescendants(), 1000);
}

/**
 * A function to return a std::vector containing pointers to each point in the tree.
 * @param tree The tree that we want to extract all of the points from.
 * @return A vector containing pointers to each point in this tree.
 */
std::vector<arma::vec*> getAllPointsInTree(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  std::vector<arma::vec*> vec;
  if (tree.NumChildren() > 0) {
    for (size_t i = 0; i < tree.NumChildren(); i++) {
      std::vector<arma::vec*> tmp = getAllPointsInTree(*(tree.Children()[i]));
      vec.insert(vec.begin(), tmp.begin(), tmp.end());
    }
  } else {
    for (size_t i = 0; i < tree.Count(); i++) {
      arma::vec* c = new arma::vec(tree.Dataset().col(tree.Points()[i]));
      vec.push_back(c);
    }
  }
  return vec;
}

// Test to ensure that none of the points in the tree are duplicates.  This,
// combined with the above test to see how many points are in the tree, should
// ensure that we inserted all points.
BOOST_AUTO_TEST_CASE(RectangleTreeConstructionRepeatTest) {
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.

  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> tree(dataset, 20, 6, 5, 2, 0);

  std::vector<arma::vec*> allPoints = getAllPointsInTree(tree);
  for (size_t i = 0; i < allPoints.size(); i++) {
    for (size_t j = i + 1; j < allPoints.size(); j++) {
      arma::vec v1 = *(allPoints[i]);
      arma::vec v2 = *(allPoints[j]);
      bool same = true;
      for (size_t k = 0; k < v1.n_rows; k++) {
        same &= (v1[k] == v2[k]);
      }
      BOOST_REQUIRE_NE(same, true);
    }
  }
  for (size_t i = 0; i < allPoints.size(); i++) {
    delete allPoints[i];
  }
}

/**
 * A function to check that each non-leaf node fully encloses its child nodes
 * and that each leaf node encloses its points.  It recurses so that it checks
 * each node under (and including) this one.
 * @param tree The tree to check.
 */
void checkContainment(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  if (tree.NumChildren() == 0) {
    for (size_t i = 0; i < tree.Count(); i++) {
      BOOST_REQUIRE_EQUAL(tree.Bound().Contains(tree.Dataset().unsafe_col(tree.Points()[i])), true);
    }
  } else {
    for (size_t i = 0; i < tree.NumChildren(); i++) {
      for (size_t j = 0; j < tree.Bound().Dim(); j++) {
        BOOST_REQUIRE_EQUAL(tree.Bound()[j].Contains(tree.Children()[i]->Bound()[j]), true);
      }
      checkContainment(*(tree.Children()[i]));
    }
  }
  return;
}

/**
 * A function to check that containment is as tight as possible.
 */
void checkExactContainment(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  if(tree.NumChildren() == 0) {
    for(size_t i = 0; i < tree.Bound().Dim(); i++) {
      double min = DBL_MAX;
      double max = -1.0 * DBL_MAX;
      for(size_t j = 0; j < tree.Count(); j++) {
	if(tree.LocalDataset().col(j)[i] < min)
	  min = tree.LocalDataset().col(j)[i];
	if(tree.LocalDataset().col(j)[i] > max)
	  max = tree.LocalDataset().col(j)[i];
      }
      BOOST_REQUIRE_EQUAL(max, tree.Bound()[i].Hi());
      BOOST_REQUIRE_EQUAL(min, tree.Bound()[i].Lo());
    }
  } else {
    for(size_t i = 0; i < tree.Bound().Dim(); i++) {
            double min = DBL_MAX;
      double max = -1.0 * DBL_MAX;
      for(size_t j = 0; j < tree.NumChildren(); j++) {
	if(tree.Child(j).Bound()[i].Lo() < min)
	  min = tree.Child(j).Bound()[i].Lo();
	if(tree.Child(j).Bound()[i].Hi() > max)
	  max = tree.Child(j).Bound()[i].Hi();
      }

      if(max != tree.Bound()[i].Hi())
	std::cout<<"error max"<<std::endl<<std::endl<<std::endl<<tree.ToString()<<std::endl<<std::endl;
      if(min != tree.Bound()[i].Lo())
	std::cout<<"error min"<<std::endl;
      BOOST_REQUIRE_EQUAL(max, tree.Bound()[i].Hi());
      BOOST_REQUIRE_EQUAL(min, tree.Bound()[i].Lo());
    }
    for(size_t i = 0; i < tree.NumChildren(); i++)
      checkExactContainment(tree.Child(i));
  }
}

/**
 * A function to check that containment is as tight as possible.
 */
void checkExactContainment(const RectangleTree<tree::RStarTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RStarTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  if(tree.NumChildren() == 0) {
    for(size_t i = 0; i < tree.Bound().Dim(); i++) {
      double min = DBL_MAX;
      double max = -1.0 * DBL_MAX;
      for(size_t j = 0; j < tree.Count(); j++) {
	if(tree.LocalDataset().col(j)[i] < min)
	  min = tree.LocalDataset().col(j)[i];
	if(tree.LocalDataset().col(j)[i] > max)
	  max = tree.LocalDataset().col(j)[i];
      }
      BOOST_REQUIRE_EQUAL(max, tree.Bound()[i].Hi());
      BOOST_REQUIRE_EQUAL(min, tree.Bound()[i].Lo());
    }
  } else {
    for(size_t i = 0; i < tree.Bound().Dim(); i++) {
      double min = DBL_MAX;
      double max = -1.0 * DBL_MAX;
      for(size_t j = 0; j < tree.NumChildren(); j++) {
	if(tree.Child(j).Bound()[i].Lo() < min)
	  min = tree.Child(j).Bound()[i].Lo();
	if(tree.Child(j).Bound()[i].Hi() > max)
	  max = tree.Child(j).Bound()[i].Hi();
      }
      BOOST_REQUIRE_EQUAL(max, tree.Bound()[i].Hi());
      BOOST_REQUIRE_EQUAL(min, tree.Bound()[i].Lo());
    }
    for(size_t i = 0; i < tree.NumChildren(); i++)
      checkExactContainment(tree.Child(i));
  }
}

/**
 * A function to check that containment is as tight as possible.
 */
void checkExactContainment(const RectangleTree<tree::XTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RStarTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  if(tree.NumChildren() == 0) {
    for(size_t i = 0; i < tree.Bound().Dim(); i++) {
      double min = DBL_MAX;
      double max = -1.0 * DBL_MAX;
      for(size_t j = 0; j < tree.Count(); j++) {
	if(tree.LocalDataset().col(j)[i] < min)
	  min = tree.LocalDataset().col(j)[i];
	if(tree.LocalDataset().col(j)[i] > max)
	  max = tree.LocalDataset().col(j)[i];
      }
      BOOST_REQUIRE_EQUAL(max, tree.Bound()[i].Hi());
      BOOST_REQUIRE_EQUAL(min, tree.Bound()[i].Lo());
    }
  } else {
    for(size_t i = 0; i < tree.Bound().Dim(); i++) {
      double min = DBL_MAX;
      double max = -1.0 * DBL_MAX;
      for(size_t j = 0; j < tree.NumChildren(); j++) {
	if(tree.Child(j).Bound()[i].Lo() < min)
	  min = tree.Child(j).Bound()[i].Lo();
	if(tree.Child(j).Bound()[i].Hi() > max)
	  max = tree.Child(j).Bound()[i].Hi();
      }
      BOOST_REQUIRE_EQUAL(max, tree.Bound()[i].Hi());
      BOOST_REQUIRE_EQUAL(min, tree.Bound()[i].Lo());
    }
    for(size_t i = 0; i < tree.NumChildren(); i++)
      checkExactContainment(tree.Child(i));
  }
}

/**
 * A function to check that parents and children are set correctly.
 */
void checkHierarchy(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
    for(size_t i = 0; i < tree.NumChildren(); i++) {
      BOOST_REQUIRE_EQUAL(&tree, tree.Child(i).Parent());
      checkHierarchy(tree.Child(i));
    }
}

/**
 * A function to check that parents and children are set correctly.
 */
void checkHierarchy(const RectangleTree<tree::RStarTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RStarTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
    for(size_t i = 0; i < tree.NumChildren(); i++) {
      BOOST_REQUIRE_EQUAL(&tree, tree.Child(i).Parent());
      checkHierarchy(tree.Child(i));
    }
}

/**
 * A function to check that parents and children are set correctly.
 */
void checkHierarchy(const RectangleTree<tree::XTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RStarTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
    for(size_t i = 0; i < tree.NumChildren(); i++) {
      BOOST_REQUIRE_EQUAL(&tree, tree.Child(i).Parent());
      checkHierarchy(tree.Child(i));
    }
}




// Test to see if the bounds of the tree are correct. (Cover all bounds and points
// beneath this node of the tree).
BOOST_AUTO_TEST_CASE(RectangleTreeContainmentTest) {
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.

  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> tree(dataset, 20, 6, 5, 2, 0);
  checkContainment(tree);
  checkExactContainment(tree);
}

/**
 * A function to ensure that the dataset for the tree, and the datasets stored
 * in each leaf node are in sync.
 * @param tree The tree to check.
 */
void checkSync(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  if (tree.IsLeaf()) {
    for (size_t i = 0; i < tree.Count(); i++) {
      for (size_t j = 0; j < tree.LocalDataset().n_rows; j++) {
        BOOST_REQUIRE_EQUAL(tree.LocalDataset().col(i)[j], tree.Dataset().col(tree.Points()[i])[j]);
      }
    }
  } else {
    for (size_t i = 0; i < tree.NumChildren(); i++) {
      checkSync(*tree.Children()[i]);
    }
  }
  return;
}

// Test to ensure that the dataset used by the whole tree (and the traversers)
// is in sync with the datasets stored in each leaf node.
BOOST_AUTO_TEST_CASE(TreeLocalDatasetInSync) {
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.

  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> tree(dataset, 20, 6, 5, 2, 0);
  checkSync(tree);
}

/**
 * A function to check that each of the fill requirements is met.  For a non-leaf node:
 * MinNumChildren() <= NumChildren() <= MaxNumChildren()
 * For a leaf node:
 * MinLeafSize() <= Count() <= MaxLeafSize
 * 
 * It recurses so that it checks each node under (and including) this one.
 * @param tree The tree to check.
 */
void checkFills(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  if (tree.IsLeaf()) {
      BOOST_REQUIRE_EQUAL((tree.Count() >= tree.MinLeafSize() || tree.Parent() == NULL) && tree.Count() <= tree.MaxLeafSize(), true);
  } else {
    for (size_t i = 0; i < tree.NumChildren(); i++) {
        BOOST_REQUIRE_EQUAL((tree.NumChildren() >= tree.MinNumChildren() || tree.Parent() == NULL) && tree.NumChildren() <= tree.MaxNumChildren(), true);
      checkFills(*tree.Children()[i]);
    }
  }
  return;
}

// Test to ensure that the minimum and maximum fills are satisfied.
BOOST_AUTO_TEST_CASE(CheckMinAndMaxFills) {
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.

  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> tree(dataset, 20, 6, 5, 2, 0);
  checkFills(tree);
}

/**
 * A function to get the height of this tree.  Though it should equal tree.TreeDepth(), we ensure
 * that every leaf node is on the same level by doing it this way.
 * @param tree The tree for which we want the height.
 * @return The height of this tree.
 */
int getMaxLevel(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  int max = 1;
  if (!tree.IsLeaf()) {
    int m = 0;
    for (size_t i = 0; i < tree.NumChildren(); i++) {
      int n = getMaxLevel(*tree.Children()[i]);
      if (n > m)
        m = n;
    }
    max += m;
  }
  return max;
}

/**
 * A function to get the "shortest height" of this tree.  Though it should equal tree.TreeDepth(), we ensure
 * that every leaf node is on the same level by doing it this way.
 * @param tree The tree for which we want the height.
 * @return The "shortest height" of the tree.
 */
int getMinLevel(const RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  int min = 1;
  if (!tree.IsLeaf()) {
    int m = INT_MAX;
    for (size_t i = 0; i < tree.NumChildren(); i++) {
      int n = getMinLevel(*tree.Children()[i]);
      if (n < m)
        m = n;
    }
    min += m;
  }
  return min;
}

// A test to ensure that all leaf nodes are stored on the same level of the tree.
BOOST_AUTO_TEST_CASE(TreeBalance) {
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.

  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> tree(dataset, 20, 6, 5, 2, 0);
  
  BOOST_REQUIRE_EQUAL(getMinLevel(tree), getMaxLevel(tree));
  BOOST_REQUIRE_EQUAL(tree.TreeDepth(), getMinLevel(tree));
}

// A test to see if point deletion is working correctly.
// We build a tree, then delete numIter points and test that the query gives correct
// results.  It is remotely possible that this test will give a false negative if
// it should happen that two points are the same distance from a third point.
BOOST_AUTO_TEST_CASE(PointDeletion) {
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.
  
  arma::mat querySet;
  querySet.randu(8, 500);

  const int numIter = 50;

  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> tree(dataset, 20, 6, 5, 2, 0);

  for (int i = 0; i < numIter; i++) {
    tree.DeletePoint(999 - i);
  }  
  
  // Do a few sanity checks.  Ensure each point is unique, the tree has the correct
  // number of points, the tree has legal containment, and the tree's data is in sync.
  std::vector<arma::vec*> allPoints = getAllPointsInTree(tree);
  for (size_t i = 0; i < allPoints.size(); i++) {
    for (size_t j = i + 1; j < allPoints.size(); j++) {
      arma::vec v1 = *(allPoints[i]);
      arma::vec v2 = *(allPoints[j]);
      bool same = true;
      for (size_t k = 0; k < v1.n_rows; k++) {
        same &= (v1[k] == v2[k]);
      }
      BOOST_REQUIRE_NE(same, true);
    }
  }
  for (size_t i = 0; i < allPoints.size(); i++) {
    delete allPoints[i];
  }
  BOOST_REQUIRE_EQUAL(tree.NumDescendants(), 1000 - numIter);
  checkContainment(tree);
  checkSync(tree);
  checkExactContainment(tree);

  mlpack::neighbor::NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
          RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> > allknn1(&tree, NULL,
          dataset, querySet, true);

  arma::Mat<size_t> neighbors1;
  arma::mat distances1;
  allknn1.Search(5, neighbors1, distances1);
  
  arma::mat newDataset;
  newDataset = dataset;
  newDataset.resize(8, 1000-numIter);
  
  arma::Mat<size_t> neighbors2;
  arma::mat distances2;
  
  // nearest neighbor search the naive way.
  mlpack::neighbor::AllkNN allknn2(newDataset, querySet,
          true, true);

  allknn2.Search(5, neighbors2, distances2);

  for (size_t i = 0; i < neighbors1.size(); i++) {
    BOOST_REQUIRE_EQUAL(distances1[i], distances2[i]);
    BOOST_REQUIRE_EQUAL(neighbors1[i], neighbors2[i]);
  }
}

// A test to see if dynamic point insertion is working correctly.
// We build a tree, then add numIter points and test that the query gives correct
// results.  It is remotely possible that this test will give a false negative if
// it should happen that two points are the same distance from a third point.
// Note that this is extremely inefficient.  You should not use dynamic insertion until
// a better solution for resizing matrices is available.
 
BOOST_AUTO_TEST_CASE(PointDynamicAdd) {
  const int numIter = 50;
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.

  RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> tree(dataset, 20, 6, 5, 2, 0);

  // Add numIter new points to the dataset.
  dataset.reshape(8, 1000+numIter);
  arma::mat tmpData;
  tmpData.randu(8, numIter);
  for (int i = 0; i < numIter; i++) {
    dataset.col(1000 + i) = tmpData.col(i);
    tree.InsertPoint(1000 + i);
  }
  
  // Do a few sanity checks.  Ensure each point is unique, the tree has the correct
  // number of points, the tree has legal containment, and the tree's data is in sync.
  std::vector<arma::vec*> allPoints = getAllPointsInTree(tree);
  for (size_t i = 0; i < allPoints.size(); i++) {
    for (size_t j = i + 1; j < allPoints.size(); j++) {
      arma::vec v1 = *(allPoints[i]);
      arma::vec v2 = *(allPoints[j]);
      bool same = true;
      for (size_t k = 0; k < v1.n_rows; k++) {
        same &= (v1[k] == v2[k]);
      }
      BOOST_REQUIRE_NE(same, true);
    }
  }
  for (size_t i = 0; i < allPoints.size(); i++) {
    delete allPoints[i];
  }
  BOOST_REQUIRE_EQUAL(tree.NumDescendants(), 1000 + numIter);
  checkContainment(tree);
  checkSync(tree);
  checkExactContainment(tree);
  
  // Now we will compare the output of the R Tree vs the output of a naive search. 
  arma::Mat<size_t> neighbors1;
  arma::mat distances1;
  arma::Mat<size_t> neighbors2;
  arma::mat distances2;
  
    // nearest neighbor search with the R tree.
  mlpack::neighbor::NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
          RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> > allknn1(&tree,
          dataset, true);

  allknn1.Search(5, neighbors1, distances1);

  // nearest neighbor search the naive way.
  mlpack::neighbor::AllkNN allknn2(dataset,
          true, true);

  allknn2.Search(5, neighbors2, distances2);

  for (size_t i = 0; i < neighbors1.size(); i++) {
    BOOST_REQUIRE_EQUAL(distances1[i], distances2[i]);
    BOOST_REQUIRE_EQUAL(neighbors1[i], neighbors2[i]);
  }  
}

/**
 * A function to check that each non-leaf node fully encloses its child nodes
 * and that each leaf node encloses its points.  It recurses so that it checks
 * each node under (and including) this one.
 * @param tree The tree to check.
 */
void checkContainment(const RectangleTree<tree::RStarTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RStarTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  if (tree.NumChildren() == 0) {
    for (size_t i = 0; i < tree.Count(); i++) {
      BOOST_REQUIRE_EQUAL(tree.Bound().Contains(tree.Dataset().unsafe_col(tree.Points()[i])), true);
    }
  } else {
    for (size_t i = 0; i < tree.NumChildren(); i++) {
      for (size_t j = 0; j < tree.Bound().Dim(); j++) {
        BOOST_REQUIRE_EQUAL(tree.Bound()[j].Contains(tree.Children()[i]->Bound()[j]), true);
      }
      checkContainment(*(tree.Children()[i]));
    }
  }
  return;
}

/**
 * A function to ensure that the dataset for the tree, and the datasets stored
 * in each leaf node are in sync.
 * @param tree The tree to check.
 */
void checkSync(const RectangleTree<tree::RStarTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RStarTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  if (tree.IsLeaf()) {
    for (size_t i = 0; i < tree.Count(); i++) {
      for (size_t j = 0; j < tree.LocalDataset().n_rows; j++) {
        BOOST_REQUIRE_EQUAL(tree.LocalDataset().col(i)[j], tree.Dataset().col(tree.Points()[i])[j]);
      }
    }
  } else {
    for (size_t i = 0; i < tree.NumChildren(); i++) {
      checkSync(*tree.Children()[i]);
    }
  }
  return;
}

/**
 * A function to ensure that the dataset for the tree, and the datasets stored
 * in each leaf node are in sync.
 * @param tree The tree to check.
 */
void checkSync(const RectangleTree<tree::XTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
        tree::RStarTreeDescentHeuristic,
        NeighborSearchStat<NearestNeighborSort>,
        arma::mat>& tree) {
  if (tree.IsLeaf()) {
    for (size_t i = 0; i < tree.Count(); i++) {
      for (size_t j = 0; j < tree.LocalDataset().n_rows; j++) {
        BOOST_REQUIRE_EQUAL(tree.LocalDataset().col(i)[j], tree.Dataset().col(tree.Points()[i])[j]);
      }
    }
  } else {
    for (size_t i = 0; i < tree.NumChildren(); i++) {
      checkSync(*tree.Children()[i]);
    }
  }
  return;
}

// A test to ensure that the SingleTreeTraverser is working correctly by comparing
// its results to the results of a naive search.
BOOST_AUTO_TEST_CASE(SingleTreeTraverserTest) {
  arma::mat dataset;
  dataset.randu(8, 1000); // 1000 points in 8 dimensions.
  arma::Mat<size_t> neighbors1;
  arma::mat distances1;
  arma::Mat<size_t> neighbors2;
  arma::mat distances2;

  RectangleTree<tree::RStarTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RStarTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> RTree(dataset, 20, 6, 5, 2, 0);

  // nearest neighbor search with the R tree.
  mlpack::neighbor::NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
          RectangleTree<tree::RStarTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RStarTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> > allknn1(&RTree,
          dataset, true);

  BOOST_REQUIRE_EQUAL(RTree.NumDescendants(), 1000);
  checkSync(RTree);
  checkContainment(RTree);
  checkExactContainment(RTree);
  checkHierarchy(RTree);

  allknn1.Search(5, neighbors1, distances1);

  // nearest neighbor search the naive way.
  mlpack::neighbor::AllkNN allknn2(dataset,
          true, true);

  allknn2.Search(5, neighbors2, distances2);

  for (size_t i = 0; i < neighbors1.size(); i++) {
    BOOST_REQUIRE_EQUAL(neighbors1[i], neighbors2[i]);
    BOOST_REQUIRE_EQUAL(distances1[i], distances2[i]);
  }
}

































// A test to ensure that the SingleTreeTraverser is working correctly by comparing
// its results to the results of a naive search.
BOOST_AUTO_TEST_CASE(XTreeTraverserTest) {
  arma::mat dataset;
  
  const int numP = 1000;
  
  dataset.randu(8, numP); // 1000 points in 8 dimensions.
  arma::Mat<size_t> neighbors1;
  arma::mat distances1;
  arma::Mat<size_t> neighbors2;
  arma::mat distances2;

  RectangleTree<tree::XTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RStarTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> RTree(dataset, 20, 6, 5, 2, 0);

  // nearest neighbor search with the R tree.
  mlpack::neighbor::NeighborSearch<NearestNeighborSort, metric::LMetric<2, true>,
          RectangleTree<tree::XTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RStarTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> > allknn1(&RTree,
          dataset, true);

  BOOST_REQUIRE_EQUAL(RTree.NumDescendants(), numP);
   checkSync(RTree);
   //checkContainment(RTree);
   checkExactContainment(RTree);
   checkHierarchy(RTree);

  allknn1.Search(5, neighbors1, distances1);

  // nearest neighbor search the naive way.
  mlpack::neighbor::AllkNN allknn2(dataset,
          true, true);

  allknn2.Search(5, neighbors2, distances2);

  for (size_t i = 0; i < neighbors1.size(); i++) {
    BOOST_REQUIRE_EQUAL(neighbors1[i], neighbors2[i]);
    BOOST_REQUIRE_EQUAL(distances1[i], distances2[i]);
  }
  
  //std::cout<<""<<RTree.ToString()<<std::endl;
}











































// Test the tree splitting.  We set MaxLeafSize and MaxNumChildren rather low
// to allow us to test by hand without adding hundreds of points.
BOOST_AUTO_TEST_CASE(RTreeSplitTest) {
  arma::mat data = arma::trans(arma::mat("0.0 0.0;"
                                         "0.0 1.0;"
                                         "1.0 0.1;"
                                         "1.0 0.5;"
                                         "0.7 0.3;"
                                         "0.9 0.9;"
                                         "0.5 0.6;"
                                         "0.6 0.3;"
                                         "0.1 0.5;"
                                         "0.3 0.7;"));
  
    RectangleTree<tree::RTreeSplit<tree::RTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> RTree(data, 5, 2, 2, 1, 0);
    
  
    //There's technically no reason they have to be in a certain order, so we
    //use firstChild etc. to arbitrarily name them.
    BOOST_REQUIRE_EQUAL(RTree.NumChildren(), 2);
    BOOST_REQUIRE_EQUAL(RTree.NumDescendants(), 10);
    BOOST_REQUIRE_EQUAL(RTree.TreeDepth(), 3);
    
    int firstChild = 0, secondChild = 1;
    if(RTree.Children()[firstChild]->NumChildren() == 2) {
      firstChild = 1;
      secondChild = 0;
    }
    
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Bound()[0].Lo(), 0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Bound()[0].Hi(), 0.1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Bound()[1].Lo(), 0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Bound()[1].Hi(), 1.0);
    
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Bound()[0].Lo(), 0.3);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Bound()[0].Hi(), 1.0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Bound()[1].Lo(), 0.1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Bound()[1].Hi(), 0.9);
    
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->NumChildren(), 1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Bound()[0].Lo(), 0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Bound()[0].Hi(), 0.1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Bound()[1].Lo(), 0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Bound()[1].Hi(), 1.0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Count(), 3);
    
    int firstPrime = 0, secondPrime = 1;
    if(RTree.Children()[secondChild]->Children()[firstPrime]->Count() == 3) {
      firstPrime = 1;
      secondPrime = 0;
    }
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->NumChildren(), 2);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Count(), 4);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Bound()[0].Lo(), 0.3);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Bound()[0].Hi(), 0.7);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Bound()[1].Lo(), 0.3);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Bound()[1].Hi(), 0.7);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Count(), 3);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Bound()[0].Lo(), 0.9);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Bound()[0].Hi(), 1.0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Bound()[1].Lo(), 0.1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Bound()[1].Hi(), 0.9);
  
}


// Test the tree splitting.  We set MaxLeafSize and MaxNumChildren rather low
// to allow us to test by hand without adding hundreds of points.
BOOST_AUTO_TEST_CASE(RStarTreeSplitTest) {
    arma::mat data = arma::trans(arma::mat("0.0 0.0;"
                                         "0.0 1.0;"
                                         "1.0 0.1;"
                                         "1.0 0.5;"
                                         "0.7 0.3;"
                                         "0.9 0.9;"
                                         "0.5 0.6;"
                                         "0.6 0.3;"
                                         "0.1 0.5;"
                                         "0.3 0.7;"));
  
    RectangleTree<tree::RStarTreeSplit<tree::RStarTreeDescentHeuristic, NeighborSearchStat<NearestNeighborSort>, arma::mat>,
          tree::RStarTreeDescentHeuristic,
          NeighborSearchStat<NearestNeighborSort>,
          arma::mat> RTree(data, 5, 2, 2, 1, 0);
	
    //There's technically no reason they have to be in a certain order, so we
    //use firstChild etc. to arbitrarily name them.
    BOOST_REQUIRE_EQUAL(RTree.NumChildren(), 2);
    BOOST_REQUIRE_EQUAL(RTree.NumDescendants(), 10);
    BOOST_REQUIRE_EQUAL(RTree.TreeDepth(), 3);
    
    int firstChild = 0, secondChild = 1;
    if(RTree.Children()[firstChild]->NumChildren() == 2) {
      firstChild = 1;
      secondChild = 0;
    }
    
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Bound()[0].Lo(), 0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Bound()[0].Hi(), 0.1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Bound()[1].Lo(), 0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Bound()[1].Hi(), 1.0);
    
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Bound()[0].Lo(), 0.3);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Bound()[0].Hi(), 1.0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Bound()[1].Lo(), 0.1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Bound()[1].Hi(), 0.9);
    
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->NumChildren(), 1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Bound()[0].Lo(), 0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Bound()[0].Hi(), 0.1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Bound()[1].Lo(), 0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Bound()[1].Hi(), 1.0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[firstChild]->Children()[0]->Count(), 3);
    
    int firstPrime = 0, secondPrime = 1;
    if(RTree.Children()[secondChild]->Children()[firstPrime]->Count() == 3) {
      firstPrime = 1;
      secondPrime = 0;
    }
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->NumChildren(), 2);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Count(), 4);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Bound()[0].Lo(), 0.3);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Bound()[0].Hi(), 0.7);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Bound()[1].Lo(), 0.3);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[firstPrime]->Bound()[1].Hi(), 0.7);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Count(), 3);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Bound()[0].Lo(), 0.9);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Bound()[0].Hi(), 1.0);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Bound()[1].Lo(), 0.1);
    BOOST_REQUIRE_EQUAL(RTree.Children()[secondChild]->Children()[secondPrime]->Bound()[1].Hi(), 0.9);
}

BOOST_AUTO_TEST_SUITE_END();
