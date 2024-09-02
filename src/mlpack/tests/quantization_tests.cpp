#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/quantization/quantization_utils.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

TEST_CASE("QuantizeLinearLayerTest", "[QuantizationTest]")
{
    LinearType<> layer(5, NoRegularizer());
    layer.Parameters().randu();

    LinearQuantization quantizer;
    arma::imat quantizedWeights = quantizer.QuantizeWeights(layer.Parameters());

    REQUIRE(quantizedWeights.min() >= -127);
    REQUIRE(quantizedWeights.max() <= 127);
}

TEST_CASE("CloneLayerTest", "[QuantizationTest]")
{
    LinearType<arma::mat, NoRegularizer> layer(4, NoRegularizer());
    layer.Parameters().randu();

    auto clonedLayer = layer.CloneAs<arma::fmat>();

    REQUIRE(clonedLayer->Parameters().n_rows == layer.Parameters().n_rows);
    REQUIRE(clonedLayer->Parameters().n_cols == layer.Parameters().n_cols);
    REQUIRE(clonedLayer->Parameters().is_finite());

    delete clonedLayer;
}
