/**
 * @file sdp.cpp
 * @author Stephen Tu
 *
 */

#include "sdp.hpp"

namespace mlpack {
namespace optimization {

SDP::SDP(const size_t n,
         const size_t numSparseConstraints,
         const size_t numDenseConstraints) :
    n(n),
    sparseC(n, n),
    denseC(n, n),
    hasModifiedSparseObjective(false),
    hasModifiedDenseObjective(false),
    sparseA(numSparseConstraints),
    sparseB(numSparseConstraints),
    denseA(numDenseConstraints),
    denseB(numDenseConstraints)
{
  denseC.zeros();
}

} // namespace optimization
} // namespace mlpack
