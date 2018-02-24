/**
 * @file selu_impl.hpp
 * @author Dakshit Agrawal
 *
 * Implementation of the SELU activation function as descibed by Gunter Klambauer,
 * Thomas Unterthiner and Andreas Mayr.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_SELU_IMPL_HPP
#define MLPACK_SELU_IMPL_HPP

#include "selu.hpp"
#include "../init_rules/gaussian_init.hpp"


namespace mlpack {
    namespace ann /** Artificial Neural Network. */ {

        //initialize weights with normal gaussian distribution having mean = 0 & variance = (1/inputSize) as mentioned in paper.
        GaussianInitialization::GaussianInitialization(const double mean, const double variance) :
            mean(0),
            variance(1/SELU::Insize()) {}



        template<typename InputDataType, typename OutputDataType>
        SELU<InputDataType, OutputDataType>::SELU(const double alpha, const double lambda, const size_t inSize,
                                                  const size_t outSize) :
                alpha(alpha),
                lambda(lambda),
                inSize(inSize),
                outSize(outSize)
        {
            weights = GaussianInitialization::Initialize(weights, inSize, outSize);
        }




        template<typename InputDataType, typename OutputDataType>
        template<typename InputType, typename OutputType>
        void SELU<InputDataType, OutputDataType>::Forward(const InputType&& input, OutputType&& output)
        {
            output = input;
            output = output*weights;
            output.each_col()+= bias;

            Fn(output, output);
        }



        template<typename InputDataType, typename OutputDataType>
        template<typename DataType>
        void SELU<InputDataType, OutputDataType>::Backward(
                const DataType&& input, DataType&& gy, DataType&& g)
        {
            DataType derivative;
            Deriv(input, derivative);
            g = gy % derivative;

            g = weights.t() * g;

        }

        template<typename InputDataType, typename OutputDataType>
        template<typename DataType>
        void SELU<InputDataType, OutputDataType>::Gradient(DataType&& input,
                                                           DataType&& error,
                                                           DataType&& gradient)
        {
            DataType derivative;
            Deriv(input, derivative);
            gradient = error % derivative;

            gradient = weights.t() * gradient;

        }

        template<typename InputDataType, typename OutputDataType>
        template<typename Archive>
        void SELU<InputDataType, OutputDataType>::serialize(
                Archive& ar,
                const unsigned int /* version */)
        {
            ar & BOOST_SERIALIZATION_NVP(alpha);
            ar & BOOST_SERIALIZATION_NVP(lambda);
            ar & BOOST_SERIALIZATION_NVP(inSize);
            ar & BOOST_SERIALIZATION_NVP(outSize);
        }

    } // namespace ann
} // namespace mlpack

#endif //MLPACK_SELU_IMPL_HPP
