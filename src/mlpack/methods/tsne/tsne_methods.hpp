/**
 * @file methods/tsne/tsne_methods.hpp
 * @author Ranjodh Singh
 *
 * Different methods which can be used to compute t-SNE gradient.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_METHODS_HPP
#define MLPACK_METHODS_TSNE_TSNE_METHODS_HPP

namespace mlpack {

//! The Exact method.
class ExactTSNE { /* Nothing to do here */ };

//! The Dual-Tree approximation method.
class DualTreeTSNE { /* Nothing to do here */ };

//! The Barnes-Hut approximation method.
class BarnesHutTSNE { /* Nothing to do here */ };

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_METHODS_HPP
