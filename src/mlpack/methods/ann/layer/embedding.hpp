/**
 * @file embedding.hpp
 * @author Manthan-R-Sheth
 *
 * Definition of the Embedding Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_EMBEDDING_HPP
#define MLPACK_EMBEDDING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <
  typename InputDataType = arma::mat,
  typename OutputDataType = arma::mat
>
class Embedding
{
  public:
    //! Create the Embedding object.
    Embedding();

    /**
     *
     * @param vocabSize Size of
     * @param dimensionSize
     * @param pretrained
     */
    Embedding(const size_t vocabSize,
              const size_t dimensionSize,
              const bool pretrain = false);
   /**
    * Reset the layer parameters.
    */
    void Reset();

    /**
     * Forward pass of the embedding layer, which forms a matrix containing
     * numeric representations of input word matrix.
     *
     * @param input Input tokenised word index matrix
     * @param output Output word representation matrix.
     */
    template<typename eT>
    void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);

    /**
     * Ordinary feed backward pass of embedding layer,calculating the function
     * f(x) by propagating x backwards through f,using the results from
     * the feed forward pass.
     *
     * @param input The propagated input activation.
     * @param gy The backpropagated error.
     * @param g The calculated gradient.
     */
    template<typename eT>
    void Backward(const arma::Mat<eT>&& /* input */,
                  arma::Mat<eT>&& gy,
                  arma::Mat<eT>&& g);

    /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
    template<typename eT>
    void Gradient(const arma::Mat<eT>&& input,
                  arma::Mat<eT>&& error,
                  arma::Mat<eT>&& gradient);

    //! Get the parameters.
    OutputDataType const& Parameters() const { return embeddingMatrix; }
    //! Modify the parameters.
    OutputDataType& Parameters() { return embeddingMatrix; }

    //! Get the input parameter.
    InputDataType const& InputParameter() const { return inputParameter; }
    //! Modify the input parameter.
    InputDataType& InputParameter() { return inputParameter; }

    //! Get the output parameter.
    OutputDataType const& OutputParameter() const { return outputParameter; }
    //! Modify the output parameter.
    OutputDataType& OutputParameter() { return outputParameter; }

    //! Get the delta.
    OutputDataType const& Delta() const { return delta; }
    //! Modify the delta.
    OutputDataType& Delta() { return delta; }

    //! Get the gradient.
    OutputDataType const& Gradient() const { return gradient; }
    //! Modify the gradient.
    OutputDataType& Gradient() { return gradient; }

    /**
     * Serialize the layer
     */
    template<typename Archive>
    void serialize(Archive& ar, const unsigned int /* version */);

  private:

    //! Locally-stored delta object.
    OutputDataType delta;

    //! Locally-stored input parameter object.
    InputDataType inputParameter;

    //! Locally-stored output parameter object.
    OutputDataType outputParameter;

    //! Locally-stored embeddingMatrix object.
    OutputDataType embeddingMatrix;

    //! Locally-stored pretrain parameter
    bool pretrain;

    //! Locally-stored vocabSize parameter
    bool vocabSize;

    //! Locally-stored dimensionSize parameter
    bool dimensionSize;

    //! Locally-stored gradient object.
    OutputDataType gradient;

}; // class Embedding
}
}

// Include implementation.
#include "embedding_impl.hpp"

#endif //MLPACK_EMBEDDING_HPP
