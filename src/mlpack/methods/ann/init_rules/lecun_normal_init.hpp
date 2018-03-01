/**
 * @file lecun_normal_init.hpp
 * @author Dakshit Agrawal
 *
 * Intialization rule given by Lecun et. al. for neural networks and
 * also mentioned in Self Normalizing Networks.
 *
 * For more information, the following papers can be referred to:
 *
 * @code
 * @inproceedings{conf/nips/KlambauerUMH17,
 * title = {Self-Normalizing Neural Networks.},
 * author = {Klambauer, Günter and Unterthiner, Thomas and Mayr, Andreas and Hochreiter, Sepp},
 * pages = {972-981},
 * year = 2017}
 *
 * @inproceedings{LeCun:1998:EB:645754.668382,
 * title = {Efficient BackProp},
 * author = {LeCun, Yann and Bottou, L{\'e}on and Orr, Genevieve B. and M\"{u}ller, Klaus-Robert},
 * year = {1998},
 * pages = {9--50}}
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_INIT_RULES_LECUN_NORMAL_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_LECUN_NORMAL_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
* This class is used to initialize weight matrix with the Lecun Normalization
* initialization rule.
*/
class LecunNormalInitialization
{
 public:
    /**
     * Initialize the LecunNormalInitialization object.
     *
     */
    LecunNormalInitialization()
    {
        // Nothing to do here.
    }

    /**
     * Initialize the elements of the weight matrix with the Lecun
     * Normal initialization rule.
     *
     * @param W Weight matrix to initialize.
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    void Initialize(arma::mat& W,
                    const size_t rows,
                    const size_t cols)
    {
        // He initialization rule says to initialize weights with random
        // values taken from a gaussian distribution with mean = 0 and
        // standard deviation = sqrt(1/rows), i.e. variance = (1/rows).
        double_t variance = 1.0 / rows;

        if (W.is_empty())
        {
            W.set_size(rows, cols);
        }

        W.imbue( [&]() { return arma::as_scalar(RandNormal(0, variance)); } );
    }

    /**
     * Initialize the elements of the specified weight 3rd order tensor
     * with Lecun Normal initialization rule.
     *
     * @param W Weight matrix to initialize.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param slice Numbers of slices.
     */
    void Initialize(arma::cube & W,
                    const size_t rows,
                    const size_t cols,
                    const size_t slices)
    {
        if (W.is_empty())
        {
            W.set_size(rows, cols, slices);
        }

        for (size_t i = 0; i < slices; i++) {
            Initialize(W.slice(i), rows, cols);
        }
    }
}; // class LecunNormalInitialization

} // namespace ann
} // namespace mlpack

#endif
