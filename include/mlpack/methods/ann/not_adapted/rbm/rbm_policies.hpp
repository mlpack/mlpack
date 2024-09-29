/**
 * @file methods/ann/rbm/rbm_policies.hpp
 * @author Shikhar Jaiswal
 *
 * Implementation of the RBM policy types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RBM_RBM_POLICIES_HPP
#define MLPACK_METHODS_ANN_RBM_RBM_POLICIES_HPP

namespace mlpack {

/**
 * For more information, see the following paper:
 *
 * @code
 * @article{Hinton10,
 *   author    = {Geoffrey Hinton},
 *   title     = {A Practical Guide to Training Restricted Boltzmann Machines},
 *   year      = {2010},
 *   url       = {https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf}
 * }
 * @endcode
 */
class BinaryRBM { /* Nothing to do here */ };

/**
 * For more information, see the following paper:
 *
 * @code
 * @article{Courville11,
 *   author  = {Aaron Courville, James Bergstra and Yoshua Bengio},
 *   title   = {A Spike and Slab Restricted Boltzmann Machine},
 *   year    = {2011},
 *   url     = {http://proceedings.mlr.press/v15/courville11a/courville11a.pdf}
 * }
 * @endcode
 */
class SpikeSlabRBM { /* Nothing to do here */ };

} // namespace mlpack

#endif
