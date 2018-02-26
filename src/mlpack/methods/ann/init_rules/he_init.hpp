/**
 * @file he_init.hpp
 * @author Dakshit Agrawal
 *
 * Intialization rule given by He et. al. for neural networks. The He
 * initialization initializes weights of the neural network to better
 * suit the rectified activation units.
 *
 * For more information, the following paper can be referred to:
 *
 * @code
 * @article{He2015DelvingDI,
 * title={Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification},
 * author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
 * journal={2015 IEEE International Conference on Computer Vision (ICCV)},
 * year={2015},
 * pages={1026-1034}}
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_INIT_RULES_HE_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_HE_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

using namespace mlpack::math;

namespace mlpack {
    namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize weight matrix with the He initialization rule.
 */
        class HeInitialization
        {
        public:
            /**
             * Initialize the HeInitialization object.
             *
             */
            HeInitialization()
            {
                // Nothing to do here.
            }

            /**
             * Initialize the elements of the weight matrix with the He initialization
             * rule.
             *
             * @param W Weight matrix to initialize.
             * @param rows Number of rows.
             * @param cols Number of columns.
             */
            void Initialize(arma::mat& W,
                            const size_t rows,
                            const size_t cols)
            {
                double_t variance = 2 / rows;

                if (W.is_empty())
                {
                    W = arma::mat(rows, cols);
                }

                W.imbue( [&]() { return arma::as_scalar(RandNormal(0, variance)); } );
            }

            /**
             * Initialize the elements of the specified weight 3rd order tensor
             * with He initialization rule.
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
                W = arma::cube(rows, cols, slices);

                for (size_t i = 0; i < slices; i++) {
                    Initialize(W.slice(i), rows, cols);
                }
            }

        }; // class HeInitialization

    } // namespace ann
} // namespace mlpack

#endif