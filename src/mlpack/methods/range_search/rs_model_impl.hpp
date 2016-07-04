/**
 * @file rs_model_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Serialize() and inline functions for RSModel.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_IMPL_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "rs_model.hpp"

namespace mlpack {
namespace range {

// Serialize the model.
template<typename Archive>
void RSModel::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(treeType, "treeType");
  ar & CreateNVP(randomBasis, "randomBasis");
  ar & CreateNVP(q, "q");

  // This should never happen, but just in case...
  if (Archive::is_loading::value)
    CleanMemory();

  // We'll only need to serialize one of the model objects, based on the type.
  switch (treeType)
  {
    case KD_TREE:
      ar & CreateNVP(kdTreeRS, "range_search_model");
      break;

    case COVER_TREE:
      ar & CreateNVP(coverTreeRS, "range_search_model");
      break;

    case R_TREE:
      ar & CreateNVP(rTreeRS, "range_search_model");
      break;

    case R_STAR_TREE:
      ar & CreateNVP(rStarTreeRS, "range_search_model");
      break;

    case BALL_TREE:
      ar & CreateNVP(ballTreeRS, "range_search_model");
      break;

    case X_TREE:
      ar & CreateNVP(xTreeRS, "range_search_model");
      break;

    case HILBERT_R_TREE:
      ar & CreateNVP(hilbertRTreeRS, "range_search_model");
      break;
  }
}

inline const arma::mat& RSModel::Dataset() const
{
  if (kdTreeRS)
    return kdTreeRS->ReferenceSet();
  else if (coverTreeRS)
    return coverTreeRS->ReferenceSet();
  else if (rTreeRS)
    return rTreeRS->ReferenceSet();
  else if (rStarTreeRS)
    return rStarTreeRS->ReferenceSet();
  else if (ballTreeRS)
    return ballTreeRS->ReferenceSet();
  else if (xTreeRS)
    return xTreeRS->ReferenceSet();
  else if (hilbertRTreeRS)
    return hilbertRTreeRS->ReferenceSet();

  throw std::runtime_error("no range search model initialized");
}

inline bool RSModel::SingleMode() const
{
  if (kdTreeRS)
    return kdTreeRS->SingleMode();
  else if (coverTreeRS)
    return coverTreeRS->SingleMode();
  else if (rTreeRS)
    return rTreeRS->SingleMode();
  else if (rStarTreeRS)
    return rStarTreeRS->SingleMode();
  else if (ballTreeRS)
    return ballTreeRS->SingleMode();
  else if (xTreeRS)
    return xTreeRS->SingleMode();
  else if (hilbertRTreeRS)
    return hilbertRTreeRS->SingleMode();

  throw std::runtime_error("no range search model initialized");
}

inline bool& RSModel::SingleMode()
{
  if (kdTreeRS)
    return kdTreeRS->SingleMode();
  else if (coverTreeRS)
    return coverTreeRS->SingleMode();
  else if (rTreeRS)
    return rTreeRS->SingleMode();
  else if (rStarTreeRS)
    return rStarTreeRS->SingleMode();
  else if (ballTreeRS)
    return ballTreeRS->SingleMode();
  else if (xTreeRS)
    return xTreeRS->SingleMode();
  else if (hilbertRTreeRS)
    return hilbertRTreeRS->SingleMode();

  throw std::runtime_error("no range search model initialized");
}

inline bool RSModel::Naive() const
{
  if (kdTreeRS)
    return kdTreeRS->Naive();
  else if (coverTreeRS)
    return coverTreeRS->Naive();
  else if (rTreeRS)
    return rTreeRS->Naive();
  else if (rStarTreeRS)
    return rStarTreeRS->Naive();
  else if (ballTreeRS)
    return ballTreeRS->Naive();
  else if (xTreeRS)
    return xTreeRS->Naive();
  else if (hilbertRTreeRS)
    return hilbertRTreeRS->Naive();

  throw std::runtime_error("no range search model initialized");
}

inline bool& RSModel::Naive()
{
  if (kdTreeRS)
    return kdTreeRS->Naive();
  else if (coverTreeRS)
    return coverTreeRS->Naive();
  else if (rTreeRS)
    return rTreeRS->Naive();
  else if (rStarTreeRS)
    return rStarTreeRS->Naive();
  else if (ballTreeRS)
    return ballTreeRS->Naive();
  else if (xTreeRS)
    return xTreeRS->Naive();
  else if (hilbertRTreeRS)
    return hilbertRTreeRS->Naive();

  throw std::runtime_error("no range search model initialized");
}

} // namespace range
} // namespace mlpack

#endif
