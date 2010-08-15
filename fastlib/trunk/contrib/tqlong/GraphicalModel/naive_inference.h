#ifndef NAIVE_INFERENCE_H
#define NAIVE_INFERENCE_H

#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/**
  * Naive inference implementation, use for testing correctness of other inference algorithm
  * It calculates the belief of each variable node in the graph by summing up all possible
  * products of factors in the graph
  */
template <typename _F>
class NaiveInference
{
public:
  typedef _F                                               factor_type;
  typedef FactorGraph<_F>                                  graph_type;
  typedef typename _F::const_iterator                      assignment_const_iterator;
  typedef typename _F::factor_value_type                   factor_value_type;
  typedef typename FactorGraph<_F>::vertex_type            vertex_type;
  typedef typename FactorGraph<_F>::vertex_vector_type     vertex_vector_type;
  typedef Map<Value, factor_value_type, ValueCompare>      belief_type;
  typedef Map<const Variable*, belief_type>                belief_map_type;
public:
  /** Constructor, preparing to make inference on a factor graph */
  NaiveInference(const graph_type& graph);

  /** The inference algorithm */
  void run();

  /** Return the result as a belief map (from variables to their beliefs) */
  belief_map_type beliefs() const { return beliefs_; }

  /** Return belief of certain variable */
  belief_type belief(const Variable* var) const;
protected:
  const graph_type& graph_;

  /** To mark visited vertex and cluster index (connected component) of each factor */
  Map<vertex_type, bool> visited_;
  Map<int, vertex_vector_type> factorClusters_;

  /** The result */
  belief_map_type beliefs_;

  /** The main calculation, summing up all possible products of factor */
  void visitFactors(const vertex_vector_type& factors, unsigned int index, factor_value_type currentVal, const Assignment& currentAsgn);

  /** Prepare the order of calculation by depth first search the graph */
  void DFSvisit(const vertex_type& u, int cluster);
  void DFSorder();

  /** Initialize and normalize the beliefs */
  void initBeliefs();
  void normalizeBeliefs();
};

template <typename _F> NaiveInference<_F>::NaiveInference(const graph_type& graph)
  : graph_(graph)
{
}

template <typename _F> void NaiveInference<_F>::run()
{
  // preparing the order of calculation
  DFSorder();
  initBeliefs();
  // visit the factors according to theirs connected components
  typedef Map<int, vertex_vector_type> map_t;
  BOOST_FOREACH(const typename map_t::value_type& p, factorClusters_)
  {
    const vertex_vector_type& factors = p.second;
    visitFactors(factors, 0, factor_value_type(1.0), Assignment());
  }
  // normalize the results
  normalizeBeliefs();
}

// Find the connected components of all factors by DFS
template <typename _F> void NaiveInference<_F>::DFSorder()
{
  const vertex_vector_type& vertices = graph_.vertices();
  for (unsigned int i = 0; i < vertices.size(); i++)
    visited_[vertices[i]] = false;
  factorClusters_.clear();
  int cluster = 0;
  BOOST_FOREACH (const vertex_type& u, vertices)
  if (!visited_[u])
  {
    DFSvisit(u, cluster);
    cluster++;
  }
}

template <typename _F> void NaiveInference<_F>::DFSvisit(const vertex_type& u, int cluster)
{
  if (u->isFactor()) factorClusters_[cluster] << u;
  visited_[u] = true;

  const vertex_vector_type& nb = graph_.neighbors(u);
  BOOST_FOREACH (const vertex_type& v, nb)
    if (!visited_[v]) DFSvisit(v, cluster);
}

template <typename _F> void NaiveInference<_F>::visitFactors(const vertex_vector_type& factors, unsigned int index,
                                                             factor_value_type currentVal, const Assignment& currentAsgn)
{
  if (index == factors.size())  // if we have the product of factors
  {
    BOOST_FOREACH(const Assignment::value_type& p, currentAsgn)
    {
      const Variable* var = p.first;
      const Value& val = p.second;
      beliefs_[var][val] += currentVal;  // add it to the belief of each variable in the assignment
    }
    return;
  }

  const factor_type& f = graph_.factor(factors[index]);
  BOOST_FOREACH (const typename factor_type::value_type& p, f) // iterate through all assignment that agrees with the current assignment
  {
    const Assignment& a = p.first;
    const factor_value_type& val = p.second;
    if (!currentAsgn.agree(a)) continue;    // only proceed if current assignmet agrees with new assignment
    Assignment newAsgn(currentAsgn);
    newAsgn.insert(a.begin(), a.end());

    visitFactors(factors, index+1, currentVal*val, newAsgn);
  }
}

// set the belief of all variables to zeros
template <typename _F> void NaiveInference<_F>::initBeliefs()
{
  const vertex_vector_type& vertices = graph_.vertices();
  BOOST_FOREACH (const vertex_type& u, vertices)
  {
    if (u->isVariable())
    {
      const Variable* var = (const Variable*) u->variable();
      for (int val = 0; val < var->cardinality(); val++)
        beliefs_[var][val] = factor_value_type(0.0);
    }
  }
}

// normalize the beliefs
template <typename _F> void NaiveInference<_F>::normalizeBeliefs()
{
  for (typename belief_map_type::iterator it = beliefs_.begin(); it != beliefs_.end(); it++)
  {
    factor_value_type sum = factor_value_type(0.0);
    belief_type& blf = (*it).second;
    for (typename belief_type::iterator bIt = blf.begin(); bIt != blf.end(); bIt++)
      sum += (*bIt).second;
    if (sum < factor_value_type(1e-15))  // sum is ZERO
    {
      for (typename belief_type::iterator bIt = blf.begin(); bIt != blf.end(); bIt++)
        (*bIt).second = factor_value_type(1.0) / factor_value_type(blf.size());
    }
    else
    {
      for (typename belief_type::iterator bIt = blf.begin(); bIt != blf.end(); bIt++)
        (*bIt).second /= sum;
    }
  }
}

template <typename _F>
    typename NaiveInference<_F>::belief_type NaiveInference<_F>::belief(const Variable* var) const
{
  typename belief_map_type::const_iterator it = beliefs_.find(var);
  DEBUG_ASSERT(it != beliefs_.end());
  return (*it).second;
}

END_GRAPHICAL_MODEL_NAMESPACE;

#endif // NAIVE_INFERENCE_H
