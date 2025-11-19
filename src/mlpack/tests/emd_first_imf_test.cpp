#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include "catch.hpp"

#include <mlpack/methods/emd/emd_first_imf.hpp>

using namespace mlpack;
using namespace arma;

TEST_CASE("FirstIMFTest", "[EMD]")
{
  vec x;
  if (!data::Load("nonstationary_signal_toy.csv", x, true))
    FAIL("Cannot load dataset");

  vec imf1;
  emd::FirstImf(x, imf1);

  REQUIRE(imf1.n_elem == x.n_elem);
  REQUIRE(mean(abs(imf1)) > 0.0);
}