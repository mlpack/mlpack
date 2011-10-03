/**
* @file emst.h
*
* @author Bill March (march@gatech.edu)
*
* This file contains utilities necessary for all of the minimum spanning tree
* algorithms.
*/

#ifndef EMST_H
#define EMST_H

#include <mlpack_core.h>

#include "union_find.h"

namespace mlpack {
namespace emst {

/**
 * An edge pair is simply two indices and a distance.  It is used as the
 * basic element of an edge list when computing a minimum spanning tree.
*/
class EdgePair {

private:
  size_t lesser_index_;
  size_t greater_index_;
  double distance_;

public:


    /**
     * Initialize an EdgePair with two indices and a distance.  The indices are
     * called lesser and greater, implying that they be sorted before calling
     * Init.  However, this is not necessary for functionality; it is just a way
     * to keep the edge list organized in other code.
     */
    void Init(size_t lesser, size_t greater, double dist) {

      mlpack::IO::AssertMessage(lesser != greater,
          "indices equal when creating EdgePair, lesser == greater");
      lesser_index_ = lesser;
      greater_index_ = greater;
      distance_ = dist;

    }

  size_t lesser_index() {
    return lesser_index_;
  }

  void set_lesser_index(size_t index) {
    lesser_index_ = index;
  }

  size_t greater_index() {
    return greater_index_;
  }

  void set_greater_index(size_t index) {
    greater_index_ = index;
  }

  double distance() const {
    return distance_;
  }

  void set_distance(double new_dist) {
    distance_ = new_dist;
  }

};// class EdgePair

}; // namespace emst
}; // namespace mlpack

#endif
