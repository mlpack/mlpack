/**
 * @file methods/pdp/pdp_impl.hpp
 * @author Ankit Singh
 *
 * Implementation of Partial Dependence Plot (PDP) for mlpack models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_PDP_PDP_IMPL_HPP
#define MLPACK_METHODS_PDP_PDP_IMPL_HPP

#include "pdp.hpp"

namespace mlpack {

    template<typename ModelType, typename Policy>
    PDP<ModelType, Policy>::PDP(const ModelType& model,
        const arma::mat& data,
        const size_t featureIndex,
        const size_t numPoints)
        : model(model),
        data(data),
        featureIndex(featureIndex),
        numPoints(numPoints)
    { /* Nothing to do. */
    }

    template<typename ModelType, typename Policy>
    std::tuple<arma::vec, arma::vec> PDP<ModelType, Policy>::Compute()
    {
        arma::vec featureValues = arma::linspace(
            arma::min(data.row(featureIndex)),
            arma::max(data.row(featureIndex)), numPoints);
        arma::vec pdpValues(numPoints, arma::fill::zeros);

        for (size_t i = 0; i < numPoints; ++i)
        {
            arma::mat modifiedData = data;
            modifiedData.row(featureIndex).fill(featureValues[i]);

            arma::rowvec predictions;
            Policy::Predict(model, modifiedData, predictions);

            pdpValues[i] = arma::mean(predictions);
        }

        return std::make_tuple(featureValues, pdpValues);
    }

} // namespace mlpack

#endif
