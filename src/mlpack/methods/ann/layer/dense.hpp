/**
 * @file dense.hpp
 * @author N Rajiv Vaidyanathan
 *
 * Definition of the Dense block class, which improves gradient and feature
 * propogation. Reduces the number of parameters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DENSE_HPP
#define MLPACK_METHODS_ANN_LAYER_DENSE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The dense block is a layer which connects each layer
 * to every other layer in a feedforward fashion. Compared to
 * traditional neural networks with L connections (connecting
 * the subsequent layers), the dense block has L(L+1)/2 connections.
 * It alleviates the vanishing-gradient problem and strengthens
 * feature propagation. It also encourages feature reuse and
 * substantially reduces the number of parameters.
 * 
 * For more information, see the following.
 * 
 * @article{DBLP:journals/corr/HuangLW16a,
 * author    = {Gao Huang and
 *              Zhuang Liu and
 *              Kilian Q. Weinberger},
 * title     = {Densely Connected Convolutional Networks},
 * journal   = {CoRR},
 * volume    = {abs/1608.06993},
 * year      = {2016},
 * url       = {http://arxiv.org/abs/1608.06993},
 * archivePrefix = {arXiv},
 * eprint    = {1608.06993},
 * timestamp = {Mon, 10 Sep 2018 15:49:32 +0200},
 * biburl    = {https://dblp.org/rec/bib/journals/corr/HuangLW16a},
 * bibsource = {dblp computer science bibliography, https://dblp.org}
 * }
 */
template<typename InputDataType = arma::mat,
         typename OutputDataType = arma::mat>
class Dense
{
}; // class Dense

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "dense_impl.hpp"

#endif
