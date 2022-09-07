/**
 * @file tests/fastmks_test.cpp
 * @author Ryan Curtin
 *
 * Ensure that fast max-kernel search is correct.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/fastmks.hpp>
#include <mlpack/methods/fastmks/fastmks_model.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace mlpack;

/**
 * Compare single-tree and naive.
 */
TEST_CASE("FastMKSSingleTreeVsNaive", "[FastMKSTest]")
{
  // First create a random dataset.
  arma::mat data;
  data.randn(5, 1000);
  LinearKernel lk;

  // Now run FastMKS naively.
  FastMKS<LinearKernel> naive(data, lk, false, true);

  arma::Mat<size_t> naiveIndices;
  arma::mat naiveProducts;
  naive.Search(10, naiveIndices, naiveProducts);

  // Now run it in single-tree mode.
  FastMKS<LinearKernel> single(data, lk, true);

  arma::Mat<size_t> singleIndices;
  arma::mat singleProducts;
  single.Search(10, singleIndices, singleProducts);

  // Compare the results.
  for (size_t q = 0; q < singleIndices.n_cols; ++q)
  {
    for (size_t r = 0; r < singleIndices.n_rows; ++r)
    {
      REQUIRE(singleIndices(r, q) == naiveIndices(r, q));
      REQUIRE(singleProducts(r, q) ==
          Approx(naiveProducts(r, q)).epsilon(1e-7));
    }
  }
}

/**
 * Compare dual-tree and naive.
 */
TEST_CASE("FastMKSDualTreeVsNaive", "[FastMKSTest]")
{
  // First create a random dataset.
  arma::mat data;
  data.randn(10, 2000);
  LinearKernel lk;

  // Now run FastMKS naively.
  FastMKS<LinearKernel> naive(data, lk, false, true);

  arma::Mat<size_t> naiveIndices;
  arma::mat naiveProducts;
  naive.Search(10, naiveIndices, naiveProducts);

  // Now run it in dual-tree mode.
  FastMKS<LinearKernel> tree(data, lk);

  arma::Mat<size_t> treeIndices;
  arma::mat treeProducts;
  tree.Search(10, treeIndices, treeProducts);

  for (size_t q = 0; q < treeIndices.n_cols; ++q)
  {
    for (size_t r = 0; r < treeIndices.n_rows; ++r)
    {
      REQUIRE(treeIndices(r, q) == naiveIndices(r, q));
      REQUIRE(treeProducts(r, q) == Approx(naiveProducts(r, q)).epsilon(1e-7));
    }
  }
}

/**
 * Compare dual-tree and single-tree on a larger dataset.
 */
TEST_CASE("DualTreeVsSingleTree", "[FastMKSTest]")
{
  // First create a random dataset.
  arma::mat data;
  data.randu(8, 2000);
  PolynomialKernel pk(5.0, 2.5);

  FastMKS<PolynomialKernel> single(data, pk, true);

  arma::Mat<size_t> singleIndices;
  arma::mat singleProducts;
  single.Search(10, singleIndices, singleProducts);

  // Now run it in dual-tree mode.
  FastMKS<PolynomialKernel> tree(data, pk);

  arma::Mat<size_t> treeIndices;
  arma::mat treeProducts;
  tree.Search(10, treeIndices, treeProducts);

  for (size_t q = 0; q < treeIndices.n_cols; ++q)
  {
    for (size_t r = 0; r < treeIndices.n_rows; ++r)
    {
      REQUIRE(treeIndices(r, q) == singleIndices(r, q));
      REQUIRE(treeProducts(r, q) ==
          Approx(singleProducts(r, q)).epsilon(1e-7));
    }
  }
}

/**
 * Test sparse FastMKS (how useful is this, I'm not sure).
 */
TEST_CASE("SparseFastMKSTest", "[FastMKSTest]")
{
  // First create a random sparse dataset.
  arma::sp_mat dataset;
  dataset.sprandu(10, 100, 0.3);

  FastMKS<LinearKernel, arma::sp_mat> sparsemks(dataset);

  arma::mat denseset(dataset);
  FastMKS<LinearKernel> densemks(denseset);

  // Store the results in these.
  arma::Mat<size_t> sparseIndices, denseIndices;
  arma::mat sparseKernels, denseKernels;

  // Do the searches.
  sparsemks.Search(3, sparseIndices, sparseKernels);
  densemks.Search(3, denseIndices, denseKernels);

  // Make sure the results are the same.
  for (size_t i = 0; i < sparseIndices.n_cols; ++i)
  {
    for (size_t j = 0; j < sparseIndices.n_rows; ++j)
    {
      if (std::abs(sparseKernels(j, i)) > 1e-15)
      {
        REQUIRE(sparseKernels(j, i) ==
            Approx(denseKernels(j, i)).epsilon(1e-7));
      }
      else
        REQUIRE(denseKernels(j, i) == Approx(0.0).margin(1e-15));
      REQUIRE(sparseIndices(j, i) == denseIndices(j, i));
    }
  }
}

TEST_CASE("SparsePolynomialFastMKSTest", "[FastMKSTest]")
{
  // Do it again with the polynomial kernel, just to be sure.
  arma::sp_mat dataset;
  dataset.sprandu(10, 100, 0.3);
  arma::mat denseset(dataset);

  PolynomialKernel pk(3);

  for (size_t i = 0; i < 100; ++i)
    for (size_t j = 0; j < 100; ++j)
    {
      if (std::abs(pk.Evaluate(dataset.col(i), dataset.col(j))) < 1e-10)
      {
        REQUIRE(pk.Evaluate(denseset.col(i), denseset.col(j)) ==
            Approx(0.0).margin(1e-10));
      }
      else
      {
        REQUIRE(pk.Evaluate(dataset.col(i), dataset.col(j)) ==
            Approx(pk.Evaluate(denseset.col(i), denseset.col(j))).
                epsilon(1e-7));
      }
    }

  FastMKS<PolynomialKernel, arma::sp_mat> sparsepoly(dataset);
  FastMKS<PolynomialKernel> densepoly(denseset);

  // Store the results in these.
  arma::Mat<size_t> sparseIndices, denseIndices;
  arma::mat sparseKernels, denseKernels;

  // Do the searches.
  sparsepoly.Search(3, sparseIndices, sparseKernels);
  densepoly.Search(3, denseIndices, denseKernels);

  // Make sure the results are the same.
  for (size_t i = 0; i < sparseIndices.n_cols; ++i)
  {
    for (size_t j = 0; j < sparseIndices.n_rows; ++j)
    {
      if (std::abs(sparseKernels(j, i)) > 1e-15)
      {
        REQUIRE(sparseKernels(j, i) ==
            Approx(denseKernels(j, i)).epsilon(1e-7));
      }
      else
        REQUIRE(denseKernels(j, i) == Approx(0.0).margin(1e-15));
      REQUIRE(sparseIndices(j, i) == denseIndices(j, i));
    }
  }
}

// Make sure the empty constructor works.
TEST_CASE("FastMKSEmptyConstructorTest", "[FastMKSTest]")
{
  FastMKS<LinearKernel> f;

  arma::mat queryData = arma::randu<arma::mat>(5, 100);
  arma::Mat<size_t> indices;
  arma::mat products;
  REQUIRE_THROWS_AS(f.Search(queryData, 3, indices, products),
      std::invalid_argument);
}

// Make sure the simplest overload of Train() works.
TEST_CASE("SimpleTrainTest", "[FastMKSTest]")
{
  arma::mat referenceSet = arma::randu<arma::mat>(5, 100);

  FastMKS<LinearKernel> f(referenceSet);
  FastMKS<LinearKernel> f2;
  f2.Train(referenceSet);

  arma::Mat<size_t> indices, indices2;
  arma::mat products, products2;

  arma::mat querySet = arma::randu<arma::mat>(5, 20);

  f.Search(querySet, 3, indices, products);
  f2.Search(querySet, 3, indices2, products2);

  REQUIRE(indices.n_rows == indices2.n_rows);
  REQUIRE(products.n_rows == products2.n_rows);
  REQUIRE(indices.n_cols == indices2.n_cols);
  REQUIRE(products.n_cols == products2.n_cols);

  for (size_t i = 0; i < products.n_elem; ++i)
  {
    if (std::abs(products[i]) < 1e-5)
      REQUIRE(products2[i] == Approx(0.0).margin(1e-5));
    else
      REQUIRE(products[i] == Approx(products2[i]).epsilon(1e-7));

    REQUIRE(indices[i] == indices2[i]);
  }
}

// Test the Train() overload that takes a kernel too.
TEST_CASE("SimpleTrainKernelTest", "[FastMKSTest]")
{
  arma::mat referenceSet = arma::randu<arma::mat>(5, 100);
  GaussianKernel gk(2.0);

  FastMKS<GaussianKernel> f(referenceSet, gk);
  FastMKS<GaussianKernel> f2;
  f2.Train(referenceSet, gk);

  arma::Mat<size_t> indices, indices2;
  arma::mat products, products2;

  arma::mat querySet = arma::randu<arma::mat>(5, 20);

  f.Search(querySet, 3, indices, products);
  f2.Search(querySet, 3, indices2, products2);

  REQUIRE(indices.n_rows == indices2.n_rows);
  REQUIRE(products.n_rows == products2.n_rows);
  REQUIRE(indices.n_cols == indices2.n_cols);
  REQUIRE(products.n_cols == products2.n_cols);

  for (size_t i = 0; i < products.n_elem; ++i)
  {
    if (std::abs(products[i]) < 1e-5)
      REQUIRE(products2[i] == Approx(0.0).margin(1e-5));
    else
      REQUIRE(products[i] == Approx(products2[i]).epsilon(1e-7));

    REQUIRE(indices[i] == indices2[i]);
  }
}

TEST_CASE("FastMKSSerializationTest", "[FastMKSTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 200);

  FastMKS<LinearKernel> f(dataset);

  FastMKS<LinearKernel> fXml, fText, fBinary;
  arma::mat otherDataset = arma::randu<arma::mat>(3, 10);
  fBinary.Train(otherDataset);

  SerializeObjectAll(f, fXml, fText, fBinary);

  arma::mat kernels, xmlKernels, jsonKernels, binaryKernels;
  arma::Mat<size_t> indices, xmlIndices, jsonIndices, binaryIndices;

  arma::mat querySet = arma::randu<arma::mat>(5, 100);

  f.Search(querySet, 5, indices, kernels);
  fXml.Search(querySet, 5, xmlIndices, xmlKernels);
  fText.Search(querySet, 5, jsonIndices, jsonKernels);
  fBinary.Search(querySet, 5, binaryIndices, binaryKernels);

  CheckMatrices(indices, xmlIndices, jsonIndices, binaryIndices);
  CheckMatrices(kernels, xmlKernels, jsonKernels, binaryKernels);
}

// Test serialization with a polynomial kernel.
TEST_CASE("PolynomialSerializationTest", "[FastMKSTest]")
{
  arma::mat dataset = arma::randu<arma::mat>(5, 200);
  PolynomialKernel* pk = new PolynomialKernel(3.0, 2.0);

  FastMKS<PolynomialKernel> f(dataset, *pk);

  arma::mat kernels, xmlKernels, jsonKernels, binaryKernels;
  arma::Mat<size_t> indices, xmlIndices, jsonIndices, binaryIndices;

  arma::mat querySet = arma::randu<arma::mat>(5, 100);
  f.Search(querySet, 5, indices, kernels);

  delete pk;

  FastMKS<PolynomialKernel> fXml, fText, fBinary;
  arma::mat otherDataset = arma::randu<arma::mat>(3, 10);
  fBinary.Train(otherDataset);

  SerializeObjectAll(f, fXml, fText, fBinary);

  fXml.Search(querySet, 5, xmlIndices, xmlKernels);
  fText.Search(querySet, 5, jsonIndices, jsonKernels);
  fBinary.Search(querySet, 5, binaryIndices, binaryKernels);

  CheckMatrices(indices, xmlIndices, jsonIndices, binaryIndices);
  CheckMatrices(kernels, xmlKernels, jsonKernels, binaryKernels);
}

// Make sure that we get an exception if we try to build the wrong FastMKSModel.
TEST_CASE("FastMKSModelWrongModelTest", "[FastMKSTest]")
{
  PolynomialKernel pk(2.0);
  arma::mat data = arma::randu<arma::mat>(5, 5);

  FastMKSModel m(FastMKSModel::LINEAR_KERNEL);
  util::Timers timers;
  REQUIRE_THROWS_AS(m.BuildModel(timers, std::move(data), pk, false, false,
      2.0), std::invalid_argument);
}

// Test the linear kernel mode of the FastMKSModel.
TEST_CASE("FastMKSModelLinearTest", "[FastMKSTest]")
{
  LinearKernel lk;
  arma::mat referenceData = arma::randu<arma::mat>(10, 100);
  arma::mat referenceCopy1(referenceData);
  arma::mat referenceCopy2(referenceData);
  arma::mat referenceCopy3(referenceData);

  FastMKS<LinearKernel> f(referenceData, lk);

  FastMKSModel m(FastMKSModel::LINEAR_KERNEL);
  FastMKSModel mNaive(FastMKSModel::LINEAR_KERNEL);
  FastMKSModel mSingle(FastMKSModel::LINEAR_KERNEL);
  util::Timers timers;

  m.BuildModel(timers, std::move(referenceCopy1), lk, false, false, 2.0);
  mNaive.BuildModel(timers, std::move(referenceCopy2), lk, false, true, 2.0);
  mSingle.BuildModel(timers, std::move(referenceCopy3), lk, true, false, 2.0);

  // Now search, first monochromatically.
  arma::Mat<size_t> indices, mIndices, mNaiveIndices, mSingleIndices;
  arma::mat kernels, mKernels, mNaiveKernels, mSingleKernels;

  f.Search(3, indices, kernels);
  m.Search(timers, 3, mIndices, mKernels);
  mNaive.Search(timers, 3, mNaiveIndices, mNaiveKernels);
  mSingle.Search(timers, 3, mSingleIndices, mSingleKernels);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }

  // Now test with a different query set.
  arma::mat querySet = arma::randu<arma::mat>(10, 50);

  f.Search(querySet, 3, indices, kernels);
  m.Search(timers, querySet, 3, mIndices, mKernels, 2.0);
  mNaive.Search(timers, querySet, 3, mNaiveIndices, mNaiveKernels, 2.0);
  mSingle.Search(timers, querySet, 3, mSingleIndices, mSingleKernels, 2.0);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }
}

// Test the polynomial kernel mode of the FastMKSModel.
TEST_CASE("FastMKSModelPolynomialTest", "[FastMKSTest]")
{
  PolynomialKernel pk(2.0);
  arma::mat referenceData = arma::randu<arma::mat>(10, 100);
  arma::mat referenceCopy1(referenceData);
  arma::mat referenceCopy2(referenceData);
  arma::mat referenceCopy3(referenceData);

  FastMKS<PolynomialKernel> f(referenceData, pk);

  FastMKSModel m(FastMKSModel::POLYNOMIAL_KERNEL);
  FastMKSModel mNaive(FastMKSModel::POLYNOMIAL_KERNEL);
  FastMKSModel mSingle(FastMKSModel::POLYNOMIAL_KERNEL);
  util::Timers timers;

  m.BuildModel(timers, std::move(referenceCopy1), pk, false, false, 2.0);
  mNaive.BuildModel(timers, std::move(referenceCopy2), pk, false, true, 2.0);
  mSingle.BuildModel(timers, std::move(referenceCopy3), pk, true, false, 2.0);

  // Now search, first monochromatically.
  arma::Mat<size_t> indices, mIndices, mNaiveIndices, mSingleIndices;
  arma::mat kernels, mKernels, mNaiveKernels, mSingleKernels;

  f.Search(3, indices, kernels);
  m.Search(timers, 3, mIndices, mKernels);
  mNaive.Search(timers, 3, mNaiveIndices, mNaiveKernels);
  mSingle.Search(timers, 3, mSingleIndices, mSingleKernels);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }

  // Now test with a different query set.
  arma::mat querySet = arma::randu<arma::mat>(10, 50);

  f.Search(querySet, 3, indices, kernels);
  m.Search(timers, querySet, 3, mIndices, mKernels, 2.0);
  mNaive.Search(timers, querySet, 3, mNaiveIndices, mNaiveKernels, 2.0);
  mSingle.Search(timers, querySet, 3, mSingleIndices, mSingleKernels, 2.0);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }
}

// Test the cosine distance mode of the FastMKSModel.
TEST_CASE("FastMKSModelCosineTest", "[FastMKSTest]")
{
  CosineDistance ck;
  arma::mat referenceData = arma::randu<arma::mat>(10, 100);
  arma::mat referenceCopy1(referenceData);
  arma::mat referenceCopy2(referenceData);
  arma::mat referenceCopy3(referenceData);

  FastMKS<CosineDistance> f(referenceData, ck);

  FastMKSModel m(FastMKSModel::COSINE_DISTANCE);
  FastMKSModel mNaive(FastMKSModel::COSINE_DISTANCE);
  FastMKSModel mSingle(FastMKSModel::COSINE_DISTANCE);
  util::Timers timers;

  m.BuildModel(timers, std::move(referenceCopy1), ck, false, false, 2.0);
  mNaive.BuildModel(timers, std::move(referenceCopy2), ck, false, true, 2.0);
  mSingle.BuildModel(timers, std::move(referenceCopy3), ck, true, false, 2.0);

  // Now search, first monochromatically.
  arma::Mat<size_t> indices, mIndices, mNaiveIndices, mSingleIndices;
  arma::mat kernels, mKernels, mNaiveKernels, mSingleKernels;

  f.Search(3, indices, kernels);
  m.Search(timers, 3, mIndices, mKernels);
  mNaive.Search(timers, 3, mNaiveIndices, mNaiveKernels);
  mSingle.Search(timers, 3, mSingleIndices, mSingleKernels);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }

  // Now test with a different query set.
  arma::mat querySet = arma::randu<arma::mat>(10, 50);

  f.Search(querySet, 3, indices, kernels);
  m.Search(timers, querySet, 3, mIndices, mKernels, 2.0);
  mNaive.Search(timers, querySet, 3, mNaiveIndices, mNaiveKernels, 2.0);
  mSingle.Search(timers, querySet, 3, mSingleIndices, mSingleKernels, 2.0);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }
}

// Test the Gaussian kernel mode of the FastMKSModel.
TEST_CASE("FastMKSModelGaussianTest", "[FastMKSTest]")
{
  GaussianKernel gk(1.5);
  arma::mat referenceData = arma::randu<arma::mat>(10, 100);
  arma::mat referenceCopy1(referenceData);
  arma::mat referenceCopy2(referenceData);
  arma::mat referenceCopy3(referenceData);

  FastMKS<GaussianKernel> f(referenceData, gk);

  FastMKSModel m(FastMKSModel::GAUSSIAN_KERNEL);
  FastMKSModel mNaive(FastMKSModel::GAUSSIAN_KERNEL);
  FastMKSModel mSingle(FastMKSModel::GAUSSIAN_KERNEL);
  util::Timers timers;

  m.BuildModel(timers, std::move(referenceCopy1), gk, false, false, 2.0);
  mNaive.BuildModel(timers, std::move(referenceCopy2), gk, false, true, 2.0);
  mSingle.BuildModel(timers, std::move(referenceCopy3), gk, true, false, 2.0);

  // Now search, first monochromatically.
  arma::Mat<size_t> indices, mIndices, mNaiveIndices, mSingleIndices;
  arma::mat kernels, mKernels, mNaiveKernels, mSingleKernels;

  f.Search(3, indices, kernels);
  m.Search(timers, 3, mIndices, mKernels);
  mNaive.Search(timers, 3, mNaiveIndices, mNaiveKernels);
  mSingle.Search(timers, 3, mSingleIndices, mSingleKernels);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }

  // Now test with a different query set.
  arma::mat querySet = arma::randu<arma::mat>(10, 50);

  f.Search(querySet, 3, indices, kernels);
  m.Search(timers, querySet, 3, mIndices, mKernels, 2.0);
  mNaive.Search(timers, querySet, 3, mNaiveIndices, mNaiveKernels, 2.0);
  mSingle.Search(timers, querySet, 3, mSingleIndices, mSingleKernels, 2.0);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }
}

// Test the Epanechnikov kernel mode of the FastMKSModel.
TEST_CASE("FastMKSModelEpanTest", "[FastMKSTest]")
{
  EpanechnikovKernel ek(2.5);
  arma::mat referenceData = arma::randu<arma::mat>(10, 100);
  arma::mat referenceCopy1(referenceData);
  arma::mat referenceCopy2(referenceData);
  arma::mat referenceCopy3(referenceData);

  FastMKS<EpanechnikovKernel> f(referenceData, ek);

  FastMKSModel m(FastMKSModel::EPANECHNIKOV_KERNEL);
  FastMKSModel mNaive(FastMKSModel::EPANECHNIKOV_KERNEL);
  FastMKSModel mSingle(FastMKSModel::EPANECHNIKOV_KERNEL);
  util::Timers timers;

  m.BuildModel(timers, std::move(referenceCopy1), ek, false, false, 2.0);
  mNaive.BuildModel(timers, std::move(referenceCopy2), ek, false, true, 2.0);
  mSingle.BuildModel(timers, std::move(referenceCopy3), ek, true, false, 2.0);

  // Now search, first monochromatically.
  arma::Mat<size_t> indices, mIndices, mNaiveIndices, mSingleIndices;
  arma::mat kernels, mKernels, mNaiveKernels, mSingleKernels;

  f.Search(3, indices, kernels);
  m.Search(timers, 3, mIndices, mKernels);
  mNaive.Search(timers, 3, mNaiveIndices, mNaiveKernels);
  mSingle.Search(timers, 3, mSingleIndices, mSingleKernels);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }

  // Now test with a different query set.
  arma::mat querySet = arma::randu<arma::mat>(10, 50);

  f.Search(querySet, 3, indices, kernels);
  m.Search(timers, querySet, 3, mIndices, mKernels, 2.0);
  mNaive.Search(timers, querySet, 3, mNaiveIndices, mNaiveKernels, 2.0);
  mSingle.Search(timers, querySet, 3, mSingleIndices, mSingleKernels, 2.0);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }
}

// Test the triangular kernel mode of the FastMKSModel.
TEST_CASE("FastMKSModelTriangularTest", "[FastMKSTest]")
{
  TriangularKernel tk(2.0);
  arma::mat referenceData = arma::randu<arma::mat>(10, 100);
  arma::mat referenceCopy1(referenceData);
  arma::mat referenceCopy2(referenceData);
  arma::mat referenceCopy3(referenceData);

  FastMKS<TriangularKernel> f(referenceData, tk);

  FastMKSModel m(FastMKSModel::TRIANGULAR_KERNEL);
  FastMKSModel mNaive(FastMKSModel::TRIANGULAR_KERNEL);
  FastMKSModel mSingle(FastMKSModel::TRIANGULAR_KERNEL);
  util::Timers timers;

  m.BuildModel(timers, std::move(referenceCopy1), tk, false, false, 2.0);
  mNaive.BuildModel(timers, std::move(referenceCopy2), tk, false, true, 2.0);
  mSingle.BuildModel(timers, std::move(referenceCopy3), tk, true, false, 2.0);

  // Now search, first monochromatically.
  arma::Mat<size_t> indices, mIndices, mNaiveIndices, mSingleIndices;
  arma::mat kernels, mKernels, mNaiveKernels, mSingleKernels;

  f.Search(3, indices, kernels);
  m.Search(timers, 3, mIndices, mKernels);
  mNaive.Search(timers, 3, mNaiveIndices, mNaiveKernels);
  mSingle.Search(timers, 3, mSingleIndices, mSingleKernels);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }

  // Now test with a different query set.
  arma::mat querySet = arma::randu<arma::mat>(10, 50);

  f.Search(querySet, 3, indices, kernels);
  m.Search(timers, querySet, 3, mIndices, mKernels, 2.0);
  mNaive.Search(timers, querySet, 3, mNaiveIndices, mNaiveKernels, 2.0);
  mSingle.Search(timers, querySet, 3, mSingleIndices, mSingleKernels, 2.0);

  REQUIRE(indices.n_cols == mIndices.n_cols);
  REQUIRE(indices.n_cols == mNaiveIndices.n_cols);
  REQUIRE(indices.n_cols == mSingleIndices.n_cols);

  REQUIRE(indices.n_rows == mIndices.n_rows);
  REQUIRE(indices.n_rows == mNaiveIndices.n_rows);
  REQUIRE(indices.n_rows == mSingleIndices.n_rows);

  REQUIRE(kernels.n_cols == mKernels.n_cols);
  REQUIRE(kernels.n_cols == mNaiveKernels.n_cols);
  REQUIRE(kernels.n_cols == mSingleKernels.n_cols);

  REQUIRE(kernels.n_rows == mKernels.n_rows);
  REQUIRE(kernels.n_rows == mNaiveKernels.n_rows);
  REQUIRE(kernels.n_rows == mSingleKernels.n_rows);

  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    REQUIRE(indices[i] == mIndices[i]);
    REQUIRE(indices[i] == mNaiveIndices[i]);
    REQUIRE(indices[i] == mSingleIndices[i]);

    if (std::abs(kernels[i]) < 1e-5)
    {
      REQUIRE(mKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mNaiveKernels[i] == Approx(0.0).margin(1e-5));
      REQUIRE(mSingleKernels[i] == Approx(0.0).margin(1e-5));
    }
    else
    {
      REQUIRE(kernels[i] == Approx(mKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mNaiveKernels[i]).epsilon(1e-7));
      REQUIRE(kernels[i] == Approx(mSingleKernels[i]).epsilon(1e-7));
    }
  }
}

TEST_CASE("FastMKSCopyConstructorTest", "[FastMKSTest]")
{
  // Create a FastMKS model, then copy it and make sure the results are valid.
  LinearKernel lk;
  arma::mat dataset = arma::randu<arma::mat>(1000, 10);

  FastMKS<LinearKernel>* f = new FastMKS<LinearKernel>(dataset, lk);

  // Copy the model.
  FastMKS<LinearKernel> newF(*f);

  // Get predictions from the first model.
  arma::Mat<size_t> indices;
  arma::mat kernels;
  f->Search(3, indices, kernels);

  // Delete the first model (exposing any memory problems) then search with the
  // second.
  delete f;

  arma::Mat<size_t> newIndices;
  arma::mat newKernels;
  newF.Search(3, newIndices, newKernels);

  REQUIRE(newIndices.n_rows == indices.n_rows);
  REQUIRE(newIndices.n_cols == indices.n_cols);
  REQUIRE(newKernels.n_rows == kernels.n_rows);
  REQUIRE(newKernels.n_cols == kernels.n_cols);

  for (size_t i = 0; i < newIndices.n_elem; ++i)
  {
    REQUIRE(newIndices[i] == indices[i]);
    if (std::abs(kernels[i]) > 1e-5)
      REQUIRE(kernels[i] == Approx(newKernels[i]).epsilon(1e-7));
    else
      REQUIRE(newKernels[i] == Approx(0.0).margin(1e-5));
  }
}

TEST_CASE("FastMKSMoveConstructorTest", "[FastMKSTest]")
{
  // Create a FastMKS object, get results, then move it and make sure the
  // results stay the same.
  LinearKernel lk;
  arma::mat dataset = arma::randu<arma::mat>(1000, 10);

  FastMKS<LinearKernel>* f = new FastMKS<LinearKernel>(dataset, lk);

  // Get predictions.
  arma::Mat<size_t> indices;
  arma::mat kernels;
  f->Search(3, indices, kernels);

  // Use the move constructor.
  FastMKS<LinearKernel> mf(std::move(*f));

  delete f;

  arma::Mat<size_t> newIndices;
  arma::mat newKernels;
  mf.Search(3, newIndices, newKernels);

  REQUIRE(newIndices.n_rows == indices.n_rows);
  REQUIRE(newIndices.n_cols == indices.n_cols);
  REQUIRE(newKernels.n_rows == kernels.n_rows);
  REQUIRE(newKernels.n_cols == kernels.n_cols);

  for (size_t i = 0; i < newIndices.n_elem; ++i)
  {
    REQUIRE(newIndices[i] == indices[i]);
    if (std::abs(kernels[i]) > 1e-5)
      REQUIRE(kernels[i] == Approx(newKernels[i]).epsilon(1e-7));
    else
      REQUIRE(newKernels[i] == Approx(0.0).margin(1e-5));
  }
}

TEST_CASE("CopyAssignmentTest", "[FastMKSTest]")
{
  // This is the same as the copy constructor test, except that it uses the
  // assignment operator.
  LinearKernel lk;
  arma::mat dataset = arma::randu<arma::mat>(1000, 10);

  FastMKS<LinearKernel>* f = new FastMKS<LinearKernel>(dataset, lk);

  // Copy the model.
  FastMKS<LinearKernel> newF = *f;

  // Get predictions from the first model.
  arma::Mat<size_t> indices;
  arma::mat kernels;
  f->Search(3, indices, kernels);

  // Delete the first model (exposing any memory problems) then search with the
  // second.
  delete f;

  arma::Mat<size_t> newIndices;
  arma::mat newKernels;
  newF.Search(3, newIndices, newKernels);

  REQUIRE(newIndices.n_rows == indices.n_rows);
  REQUIRE(newIndices.n_cols == indices.n_cols);
  REQUIRE(newKernels.n_rows == kernels.n_rows);
  REQUIRE(newKernels.n_cols == kernels.n_cols);

  for (size_t i = 0; i < newIndices.n_elem; ++i)
  {
    REQUIRE(newIndices[i] == indices[i]);
    if (std::abs(kernels[i]) > 1e-5)
      REQUIRE(kernels[i] == Approx(newKernels[i]).epsilon(1e-7));
    else
      REQUIRE(newKernels[i] == Approx(0.0).margin(1e-5));
  }
}
