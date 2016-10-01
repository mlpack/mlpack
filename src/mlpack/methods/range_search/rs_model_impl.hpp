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
#include <boost/serialization/variant.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::range;


namespace mlpack {
namespace range {


//! Monochromatic range search on the given RSType instance.
template<typename RSType>
void MonoSearchVisitor::operator()(RSType *rs) const
{
    if(rs)
        return rs->Search(range, neighbors, distances);
    throw std::runtime_error("no range search model initialized");
}

//! Save parameters for the bichromatic range search.
BiSearchVisitor::BiSearchVisitor(arma::mat& querySet,
                                const math::Range& range,
                                std::vector<std::vector<size_t>>& neighbors,
                                std::vector<std::vector<double>>& distances,
                                const size_t leafSize) :
    querySet(querySet),
    range(range),
    neighbors(neighbors),
    distances(distances),
    leafSize(leafSize);
{}

//! Default Bichromatic Range Search for the given RSType instance.
template<template<typename TreeMetricType,
              typename TreeStatType,
              typename TreeMatType> class TreeType>
void BiSearchVisitor::operator()(RSTypeT<TreeType>* rs) const
{
    if(rs)
        return rs->Search(querySet, range, neighbors, distances);
    throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search for the given RSType specialized for KDTrees.
void BiSearchVisitor::operator()(RSTypeT<tree::KDTree>* rs) const
{
    if(rs)
        return SearchLeaf(rs);
    throw std::runtime_error("no range search model initialized");
}

//! Bichromatic Range Search for the given RSType specialized for BallTrees.
void BiSearchVisitor::operator()(RSTypeT<tree::BallTree>* rs) const
{
    if(rs)
        return SearchLeaf(rs);
    throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search on the given RSType considering the leafSize.
template<typename RSType>
void BiSearchVisitor::SearchLeaf(RSType* rs) const
{
    if(!rs->Naive() && !rs->SingleMode())
    {
        // Build a second tree and search.
        vector<size_t> oldFromNewQueries;
        RSType::Tree queryTree(move(querySet), oldFromNewQueries,
            leafSize);

        vector<vector<size_t>> neighborsOut;
        vector<vector<double>> distancesOut;
        rs->Search(&queryTree, range, neighborsOut, distancesOut);

        // Remap the query points.
        neighbors.resize(queryTree.Dataset().n_cols);
        distances.resize(queryTree.Dataset().n_cols);
        for (size_t i = 0; i < queryTree.Dataset().n_cols; ++i)
        {
          neighbors[oldFromNewQueries[i]] = neighborsOut[i];
          distances[oldFromNewQueries[i]] = distancesOut[i];
        }

    }
    else
    {
        // Search without building a second tree.
        rs->Search(querySet, range, neighbors, distances);
    }
}

//! Expose the referenceSet of the given RSType.
template<typename RSType>
const arma::mat& ReferenceSetVisitor::operator()(RSType* rs) const
{
    if(rs)
        return rs->ReferenceSet();
    throw std::runtime_error("no range search model initialized");
}

//! Return whether in single-tree search mode.
template<typename RSType>
bool SingleModeVisitor::operator()(RSType *rs) const
{
    if(rs)
        return rs->SingleMode();
    throw std::runtime_error("no range search model initialized");
}

//! Modify whether in single search mode.
template<typename RSType>
bool& SetSingleModeVisitor::operator()(RSType *rs)
{
    if(rs)
        return rs->SingleMode();
    throw std::runtime_error("no range search model initialized");
}

//! Return whether in naive search mode.
template<typename RSType>
bool NaiveVisitor::operator()(RSType *rs)
{
    if(rs)
        return rs->Naive();
    throw std::runtime_error("No range search model initialized");
}

//! Modify whether in naive search mode.
template<typename RSType>
bool& SetNaiveVisitor::operator()(RSType *rs)
{
    if(rs)
        return rs->Naive();
    throw std::runtime_error("No range search model initialized");
}

//! Clean memory, if necessary.
template<typename RSType>
void DeleteVisitor::operator()(RSType* rs) const
{
    if(rs)
        delete rs;
}



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
    boost::apply_visitor(DeleteVisitor(), rSearch);
  
  
  const std::string& name = "range_search_model";
  ar & CreateNVP(rSearch, name);
}

//! Expose the dataset. 
inline const arma::mat& RSModel::Dataset() const
{
    return boost::apply_visitor(ReferenceSetVisitor(), rSearch);
}

//! Get whether in single tree search mode.
inline bool RSModel::SingleMode() const
{
    return boost::apply_visitor(SingleModeVisitor(), rSearch);
}

//! Modify whether in single tree search mode.
inline bool& RSModel::SingleMode()
{
    return boost::apply_visitor(SetSingleModeVisitor(), rSearch);
}

//! Get whether in naive search mode.
inline bool RSModel::Naive() const
{
    return boost::apply_visitor(NaiveVisitor(), rSearch);
}

//! Modify whether in naive search mode.
inline bool& RSModel::Naive()
{
    return boost::apply_visitor(SetNaiveVisitor(), rSearch);
}

} // namespace range
} // namespace mlpack

#endif
