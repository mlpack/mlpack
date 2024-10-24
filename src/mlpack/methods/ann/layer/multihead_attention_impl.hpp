/**
 * @file methods/ann/layer/multihead_attention_impl.hpp
 * @author Mrityunjay Tripathi
 * @author Adam Kropp

 * Implementation of the MultiheadAttention class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_MULTIHEAD_ATTENTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIHEAD_ATTENTION_IMPL_HPP

// In case it hasn't yet been included.
#include "multihead_attention.hpp"

#include <mlpack/core/math/multiply_slices.hpp>

namespace mlpack {

template <typename MatType, typename RegularizerType>
MultiheadAttentionType<MatType, RegularizerType>::
MultiheadAttentionType() :
    tgtSeqLen(0),
    srcSeqLen(0),
    embedDim(0),
    numHeads(0),
    headDim(0),
    selfAttention(false)
{
  // Nothing to do here.
}

template <typename MatType, typename RegularizerType>
MultiheadAttentionType<MatType, RegularizerType>::
MultiheadAttentionType(
    const size_t tgtSeqLen,
    const size_t numHeads,
    const MatType& attnmask,
    const MatType& keypaddingmask,
    const bool selfAttention) :
    tgtSeqLen(tgtSeqLen),
    srcSeqLen(0),
    embedDim(0),
    numHeads(numHeads),
    attnMask(attnmask),
    keyPaddingMask(keypaddingmask),
    selfAttention(selfAttention)
{
}

template <typename MatType, typename RegularizerType>
void MultiheadAttentionType<MatType, RegularizerType>::SetWeights(
    const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, (4 * embedDim + 4) * embedDim, 1);

  MakeAlias(queryWt, weightsIn, embedDim, embedDim);
  MakeAlias(keyWt, weightsIn, embedDim, embedDim, embedDim * embedDim);
  MakeAlias(valueWt, weightsIn, embedDim, embedDim, 2 * embedDim * embedDim);
  MakeAlias(outWt, weightsIn, embedDim, embedDim, 3 * embedDim * embedDim);

  MakeAlias(qBias, weightsIn, embedDim, 1, 4 * embedDim * embedDim);
  MakeAlias(kBias, weightsIn, embedDim, 1, (4 * embedDim + 1) * embedDim);
  MakeAlias(vBias, weightsIn, embedDim, 1, (4 * embedDim + 2) * embedDim);
  MakeAlias(outBias, weightsIn, 1, embedDim, (4 * embedDim + 3) * embedDim);
}

template <typename MatType, typename RegularizerType>
void MultiheadAttentionType<MatType, RegularizerType>::
Forward(const MatType& input, MatType& output)
{
  using CubeType = arma::Cube<typename MatType::elem_type>;

  if (input.n_rows != embedDim *
      (selfAttention ? srcSeqLen : (tgtSeqLen + 2 * srcSeqLen)))
  {
    Log::Fatal << "Incorrect input dimensions!" << std::endl;
  }

  if (selfAttention && tgtSeqLen != srcSeqLen)
  {
    Log::Fatal << "Target sequence length (" << tgtSeqLen << ") and source "
        << "sequence length (" << srcSeqLen << ") must match when using "
        << "self-attention!" << std::endl;
  }

  const size_t batchSize = input.n_cols;

  // shape of output : (embedDim * tgtSeqLen, batchSize).
  output.set_size(embedDim * tgtSeqLen, batchSize);

  // Reshape the input, the query, and the key into a cube from a matrix.
  // The shape of q : (embedDim, tgtSeqLen, batchSize).
  // The shape of k : (embedDim, srcSeqLen, batchSize).
  // The shape of v : (embedDim, srcSeqLen, batchSize).
  const CubeType q(const_cast<MatType&>(input).memptr(),
      embedDim, tgtSeqLen, batchSize, false, false);
  const CubeType k(const_cast<MatType&>(input).memptr() +
      (selfAttention ? 0 : embedDim * tgtSeqLen * batchSize),
      embedDim, srcSeqLen, batchSize, false, false);
  const CubeType v(const_cast<MatType&>(input).memptr() +
      (selfAttention ? 0 : embedDim * (tgtSeqLen + srcSeqLen) * batchSize),
      embedDim, srcSeqLen, batchSize, false, false);

  // qProj, kProj, and vProj are the linearly projected query, key and value
  // respectively.
  qProj.set_size(tgtSeqLen, embedDim, batchSize);
  kProj.set_size(srcSeqLen, embedDim, batchSize);
  vProj.set_size(srcSeqLen, embedDim, batchSize);

  for (size_t i = 0; i < batchSize; ++i)
  {
    qProj.slice(i) = trans(
        queryWt * q.slice(i) + repmat(qBias, 1, tgtSeqLen));
    kProj.slice(i) = trans(
        keyWt * k.slice(i) + repmat(kBias, 1, srcSeqLen));
    vProj.slice(i) = trans(
        valueWt * v.slice(i) + repmat(vBias, 1, srcSeqLen));
  }

  // The scaling factor sqrt(headDim) is used to prevent exploding values
  // after dot product i.e. when qProj is multiplied with kProj.
  qProj /= std::sqrt(headDim);

  // Split the qProj, kProj and vProj into n heads. That's what Multihead
  // Attention is.
  qProj.reshape(tgtSeqLen, headDim, numHeads * batchSize);
  kProj.reshape(srcSeqLen, headDim, numHeads * batchSize);
  vProj.reshape(srcSeqLen, headDim, numHeads * batchSize);

  // Calculate the scores i.e. perform the matrix multiplication operation
  // on qProj and kProj. Here score = qProj . kProj'
  scores = MultiplyCube2Cube(qProj, kProj, false, true);

  // Apply the attention mask if provided. The attention mask is used to black-
  // out future sequences and generally used in Encoder-Decoder attention.
  // The attention mask has elements -inf or 0.
  // The shape of the attention mask : (tgtSeqLen, srcSeqLen).
  if (!attnMask.is_empty())
  {
    if (attnMask.n_rows != tgtSeqLen || attnMask.n_cols != srcSeqLen)
      Log::Fatal << "The size of the 'attn_mask' is not correct.\n";
    scores.each_slice() += attnMask;
  }

  // Apply the key padding mask when provided. It blacks-out any particular
  // word in the sequence.
  // The key padding mask has elements -inf or 0
  // The shape of keyPaddingMask : (1, srcSeqLen).
  if (!keyPaddingMask.is_empty())
  {
    if (keyPaddingMask.n_rows != 1 || keyPaddingMask.n_cols != srcSeqLen)
        Log::Fatal << "The size of the 'keyPaddingMask' is not correct.\n";
    scores.each_slice() += repmat(keyPaddingMask, tgtSeqLen, 1);
  }

  for (size_t i = 0; i < numHeads * batchSize; ++i)
  {
    softmax.Forward(scores.slice(i), scores.slice(i));
  }

  // Calculate the attention output i.e. matrix multiplication of softmax
  // output and vProj.
  // The shape of attnOutput : (tgtSeqLen, headDim, numHeads * batchSize).
  attnOut = MultiplyCube2Cube(scores, vProj, false, false);

  // Now we will concatenate output of all the heads i.e. we will reshape
  // attnOut to (tgtSeqLen, embedDim, batchSize).
  attnOut.reshape(tgtSeqLen, embedDim, batchSize);

  // The final output is the linear projection of attention output.
  for (size_t i = 0; i < batchSize; ++i)
  {
    output.col(i) = vectorise(trans(attnOut.slice(i) * outWt
        + repmat(outBias, tgtSeqLen, 1)));
  }
}

template <typename MatType, typename RegularizerType>
void MultiheadAttentionType<MatType, RegularizerType>::
Backward(const MatType& /* input */,
         const MatType& /* output */,
         const MatType& gy,
         MatType& g)
{
  using CubeType = arma::Cube<typename MatType::elem_type>;

  if (gy.n_rows != tgtSeqLen * embedDim)
  {
    Log::Fatal << "Backpropagated error has incorrect dimensions!" << std::endl;
  }

  const size_t batchSize = gy.n_cols;
  g.set_size(selfAttention ? (embedDim * srcSeqLen) :
      embedDim * (tgtSeqLen + 2 * srcSeqLen), batchSize);

  // Reshape the propagated gradient into a cube.
  // The shape of gyTemp : (tgtSeqLen, embedDim, batchSize).
  // We need not split it into n heads now because this is the part when
  // output were concatenated from n heads.
  CubeType gyTemp(const_cast<MatType&>(gy).memptr(), embedDim,
      tgtSeqLen, batchSize, true, false);

  // The shape of gyTemp : (embedDim, tgtSeqLen, batchSize).
  // The shape of outWt : (embedDim, embedDim).
  // The shape of the result : (tgtSeqLen, embedDim, batchSize).
  gyTemp = MultiplyCube2Mat(gyTemp, outWt, true, true);

  // Now since the shape of gyTemp is (tgtSeqLen, embedDim, batchSize). We will
  // split it into n heads.
  // The shape of gyTemp : (tgtSeqLen, headDim, numHeads * batchSize).
  gyTemp.reshape(tgtSeqLen, headDim, numHeads * batchSize);

  // Obtain backpropagted error of value.
  // Shape of gyTemp : (tgtSeqLen, headDim, numHeads * batchSize).
  // Shape of scores : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
  // The shape of tmp : (srcSeqLen, headDim, numHeads * batchSize).
  CubeType tmp = MultiplyCube2Cube(scores, gyTemp, true, false);

  // Concatenate results of all the attention heads.
  tmp.reshape(srcSeqLen, embedDim, batchSize);

  for (size_t i = 0; i < batchSize; ++i)
  {
    if (selfAttention)
    {
      g.submat(0, i, g.n_rows - 1, i) =
          vectorise(trans(tmp.slice(i) * valueWt));
    }
    else
    {
      g.submat((tgtSeqLen + srcSeqLen) * embedDim, i, g.n_rows - 1, i) =
          vectorise(trans(tmp.slice(i) * valueWt));
    }
  }

  // The shape of gyTemp : (tgtSeqLen, headDim, numHeads * batchSize).
  // The shape of vProj : (srcSeqLen, headDim, numHeads * batchSize).
  // So the new shape of gyTemp : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
  gyTemp = MultiplyCube2Cube(gyTemp, vProj, false, true);

  for (size_t i = 0; i < numHeads * batchSize; ++i)
  {
    // We will perform backpropagation of softmax over each slice of gyTemp.
    softmax.Backward({} /* unused */, scores.slice(i), gyTemp.slice(i),
        gyTemp.slice(i));
  }

  // Obtain backpropagated error of key.
  // The shape of qProj : (tgtSeqLen, headDim, numHeads * batchSize).
  // The shape of gyTemp : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
  // The new shape of tmp : (srcSeqLen, headDim, numHeads * batchSize).
  tmp = MultiplyCube2Cube(gyTemp, qProj, true, false);

  // Concatenate results of all the attention heads.
  tmp.reshape(srcSeqLen, embedDim, batchSize);

  for (size_t i = 0; i < batchSize; ++i)
  {
    if (selfAttention)
    {
      // Sum the query, key, and value deltas.
      g.submat(0, i, g.n_rows - 1, i) +=
          vectorise(trans(tmp.slice(i) * keyWt));
    }
    else
    {
      g.submat(tgtSeqLen * embedDim, i,
               (tgtSeqLen + srcSeqLen) * embedDim - 1, i) =
          vectorise(trans(tmp.slice(i) * keyWt));
    }
  }

  // Obtain backpropagated error of the query.
  // The shape of kProj : (srcSeqLen, headDim, numHeads * batchSize).
  // The shape of gyTemp : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
  // The new shape of tmp : (tgtSeqLen, headDim, numHeads * batchSize).
  tmp = MultiplyCube2Cube(gyTemp, kProj) / std::sqrt(headDim);

  // Concatenate results of all the attention heads.
  tmp.reshape(tgtSeqLen, embedDim, batchSize);

  for (size_t i = 0; i < batchSize; ++i)
  {
    if (selfAttention)
    {
      // Sum the query, key, and value deltas.
      g.submat(0, i, g.n_rows - 1, i) +=
          vectorise(trans(tmp.slice(i) * queryWt));
    }
    else
    {
      g.submat(0, i, tgtSeqLen * embedDim - 1, i) =
          vectorise(trans(tmp.slice(i) * queryWt));
    }
  }
}

template <typename MatType, typename RegularizerType>
void MultiheadAttentionType<MatType, RegularizerType>::
Gradient(const MatType& input,
         const MatType& error,
         MatType& gradient)
{
  using CubeType = arma::Cube<typename MatType::elem_type>;

  if (input.n_rows != embedDim * (selfAttention ? srcSeqLen :
      (tgtSeqLen + 2 * srcSeqLen)))
  {
    Log::Fatal << "Incorrect input dimensions!" << std::endl;
  }

  if (selfAttention && tgtSeqLen != srcSeqLen)
  {
    Log::Fatal << "Target sequence length (" << tgtSeqLen << ") and source "
        << "sequence length (" << srcSeqLen << ") must match when using "
        << "self-attention!" << std::endl;
  }

  if (error.n_rows != tgtSeqLen * embedDim)
  {
    Log::Fatal << "Backpropagated error has incorrect dimensions." << std::endl;
  }

  const size_t batchSize = input.n_cols;
  const size_t wtSize = embedDim * embedDim;

  // The shape of gradient : (4 * embedDim * embedDim + 4 * embedDim, 1).
  gradient.set_size(arma::size(weights));

  const CubeType q(const_cast<MatType&>(input).memptr(),
      embedDim, tgtSeqLen, batchSize, false, false);
  const CubeType k(const_cast<MatType&>(input).memptr() +
      (selfAttention ? 0 : q.n_elem), embedDim, srcSeqLen, batchSize, false,
      false);
  const CubeType v(const_cast<MatType&>(input).memptr() +
      (selfAttention ? 0 : (q.n_elem + k.n_elem)), embedDim, srcSeqLen,
      batchSize, false, false);

  // Reshape the propagated error into a cube.
  // The shape of errorTemp : (embedDim, tgtSeqLen, batchSize).
  CubeType errorTemp(const_cast<MatType&>(error).memptr(), embedDim,
      tgtSeqLen, batchSize, true, false);

  // Gradient wrt. outBias, i.e. dL/d(outBias).
  gradient.rows(4 * wtSize + 3 * embedDim, 4 * wtSize + 4 * embedDim - 1)
      = vectorise(sum(sum(errorTemp, 2), 1));

  // The shape of attnOut : (tgtSeqLen, embedDim, batchSize).
  // The shape of errorTemp : (embedDim, tgtSeqLen, batchSize).
  // The shape of gyTemp : (embedDim, embedDim, batchSize).
  CubeType gyTemp = MultiplyCube2Cube(attnOut, errorTemp, true, true);

  // Gradient wrt. outWt, i.e. dL/d(outWt). We will take sum of gyTemp along
  // the slices and vectorise the output.
  gradient.rows(3 * wtSize, 4 * wtSize - 1) = vectorise(sum(gyTemp, 2));

  // Partial derivative wrt. attnOut.
  // The shape of outWt : (embedDim, embedDim).
  // The shape of errorTemp : (embedDim, tgtSeqLen, batchSize).
  // The shape of gyTemp : (tgtSeqLen, embedDim, batchSize).
  gyTemp = MultiplyCube2Mat(errorTemp, outWt, true, true);

  // Now we will split it into n heads i.e. reshape it into a cube of shape
  // (tgtSeqLen, headDim, numHeads * batchSize).
  gyTemp.reshape(tgtSeqLen, headDim, numHeads * batchSize);

  // Shape of gyTemp : (tgtSeqLen, headDim, numHeads * batchSize).
  // Shape of scores : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
  // The new shape of errorTemp : (srcSeqLen, headDim, numHeads * batchSize).
  errorTemp = MultiplyCube2Cube(scores, gyTemp, true, false);

  // Now we will concatenate the propagated errors from all heads i.e. we
  // will reshape errorTemp to (srcSeqLen, embedDim, batchSize).
  errorTemp.reshape(srcSeqLen, embedDim, batchSize);

  // Gradient wrt. vBias, i.e. dL/d(vBias). We will take summation of errorTemp
  // over all the batches and over all the sequences.
  gradient.rows(4 * wtSize + 2 * embedDim, 4 * wtSize + 3 * embedDim - 1)
      = vectorise(sum(sum(errorTemp, 2), 0));

  // Shape of v : (srcSeqLen, embedDim, batchSize).
  // Shape of errorTemp : (srcSeqLen, embedDim, bathSize).
  // The new shape of errorTemp : (embedDim, embedDim, batchSize).
  errorTemp = MultiplyCube2Cube(errorTemp, v, true, true);

  // Gradient wrt. valueWt, i.e. dL/d(valueWt). We will take summation over all
  // batches of errorTemp.
  gradient.rows(2 * wtSize, 3 * wtSize - 1) = vectorise(sum(errorTemp, 2));

  // Now, the shape of gyTemp : (tgtSeqLen, headDim, numHeads * batchSize).
  // The shape of vProj : (srcSeqLen, headDim, numHeads * batchSize).
  // The new shape of errorTemp : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
  errorTemp = MultiplyCube2Cube(gyTemp, vProj, false, true);

  for (size_t i = 0; i < numHeads * batchSize; ++i)
  {
    // The shape of scores : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
    // The shape of errorTemp : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
    // The new shape of errorTemp remain same.
    softmax.Backward({} /* unused */, scores.slice(i), errorTemp.slice(i),
        errorTemp.slice(i));
  }

  // The shape of qProj : (tgtSeqLen, headDim, numHeads * batchSize).
  // The shape of errorTemp : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
  // The shape of gyTemp : (srcSeqLen, headDim, numHeads * batchSize).
  gyTemp = MultiplyCube2Cube(errorTemp, qProj, true, false);

  // We will now conctenate the propagated errors from all heads.
  // The new shape of gyTemp : (srcSeqLen, embedDim, batchSize).
  gyTemp.reshape(srcSeqLen, embedDim, batchSize);

  // Gradient wrt. kBias, i.e. dL/d(kBias). We will take summation over all the
  // batches of gyTemp and then over all the sequences.
  gradient.rows(4 * wtSize + embedDim, 4 * wtSize + 2 * embedDim - 1)
      = vectorise(sum(sum(gyTemp, 2), 0));

  // The shape of k : (embedDim, srcSeqLen, batchSize).
  // The shape of gyTemp : (srcSeqLen, embedDim, batchSize).
  // The shape of dkeyWt : (embedDim, embedDim, batchSize).
  gyTemp = MultiplyCube2Cube(gyTemp, k, true, true);

  // Gradient wrt. keyWt, i.e. dL/d(keyWt). We will take summation over all the
  // batches of dkeyWt.
  gradient.rows(wtSize, 2 * wtSize - 1) = vectorise(sum(gyTemp, 2));

  // The shape of kProj : (srcSeqLen, headDim, numHeads * batchSize).
  // The shape of errorTemp : (tgtSeqLen, srcSeqLen, numHeads * batchSize).
  // The shape of gyTemp : (tgtSeqLen, headDim, numHeads * batchSize).
  gyTemp = MultiplyCube2Cube(errorTemp, kProj, false, false);

  // Now, we will concatenate propagated error of all heads.
  gyTemp.reshape(tgtSeqLen, embedDim, batchSize);
  gyTemp /= std::sqrt(headDim);

  // Gradient wrt. qBias, i.e. dL/d(qBias). We will take summation over all the
  // batches of gyTemp and over all the sequences.
  gradient.rows(4 * wtSize, 4 * wtSize + embedDim - 1)
      = vectorise(sum(sum(gyTemp, 2), 0));

  // The shape of gyTemp : (tgtSeqLen, embedDim, batchSize).
  // The shape of q : (embedDim, tgtSeqLen, batchSize).
  // The shape of gyTemp : (embedDim, embedDim, batchSize).
  gyTemp = MultiplyCube2Cube(gyTemp, q, true, true);

  // Gradient wrt. queryWt, i.e. dL/d(queryBias). We will take summation over
  // all the batches of gyTemp.
  gradient.rows(0, wtSize - 1) = vectorise(sum(gyTemp, 2));

  // Regularize according to the given regularization rule.
  regularizer.Evaluate(weights, gradient);
}

template <typename MatType, typename RegularizerType>
template <typename Archive>
void MultiheadAttentionType<MatType, RegularizerType>::
serialize(Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(tgtSeqLen));
  ar(CEREAL_NVP(srcSeqLen));
  ar(CEREAL_NVP(embedDim));
  ar(CEREAL_NVP(numHeads));
  ar(CEREAL_NVP(headDim));
  ar(CEREAL_NVP(selfAttention));
  ar(CEREAL_NVP(softmax));
  ar(CEREAL_NVP(regularizer));
  ar(CEREAL_NVP(attnMask));
  ar(CEREAL_NVP(keyPaddingMask));

  if (Archive::is_loading::value)
  {
    queryWt.clear();
    keyWt.clear();
    valueWt.clear();
    outWt.clear();
    qBias.clear();
    kBias.clear();
    vBias.clear();
    outBias.clear();
    weights.clear();
    qProj.clear();
    kProj.clear();
    vProj.clear();
    scores.clear();
    attnOut.clear();
  }
}

} // namespace mlpack

#endif
