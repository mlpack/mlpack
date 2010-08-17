#ifndef NAIVE_INFERENCE_H
#define NAIVE_INFERENCE_H

#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** Naive inference implementation, use for testing correctness of other inference algorithms.
  * It calculates the belief of each variable node in the graph by summing up all possible
  * products of factors in the graph.
  * It may be slow but it is the exact inference algorithm.
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
  typedef Map<vertex_type, belief_type>                    belief_map_type;
public:
  /** Constructor, preparing to make inference on a factor graph */
  NaiveInference(const graph_type& graph);

  /** The inference algorithm */
  void run();

  /** Return the result as a belief map (from variables to their beliefs) */
  const belief_map_type& beliefs() const { return beliefs_; }
  const graph_type& graph() const { return graph_; }

  /** Return belief of certain variable */
  const belief_type& belief(const vertex_type& v) const;
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
  cout << "----------------------- Naive Inference ---------------------------------" << endl;
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
  BOOST_FOREACH (const vertex_type& u, vertices)
    visited_[u] = false;
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
    // Update the belief of variables
    BOOST_FOREACH(const Assignment::value_type& p, currentAsgn)
    {
      const Variable* var = p.first;
      const Value& val = p.second;
      const vertex_type& u = graph_.dataVertexMap().get((void*) var);
      beliefs_[u][val] += currentVal;  // add it to the belief of each variable in the assignment
    }
    // Update the mean value of factors
    BOOST_FOREACH(const vertex_type& u, factors)
    {
      const factor_value_type& f_val = graph_.factor(u).get(currentAsgn);  // u.factor
      // currentVal is the product of all factors
      beliefs_[u][0] += f_val*currentVal;
      beliefs_[u][1] += currentVal;
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
        beliefs_[u][val] = factor_value_type(0.0);
    }
    else  // u is a factor
    {
      beliefs_[u][0] = factor_value_type(0.0);   // the sum of  u.factor * (product of all factors)
      beliefs_[u][1] = factor_value_type(0.0);   // the sum of products of all factors
    }
  }
}

// normalize the beliefs
template <typename _F> void NaiveInference<_F>::normalizeBeliefs()
{
  for (typename belief_map_type::iterator it = beliefs_.begin(); it != beliefs_.end(); it++)
  {
    const vertex_type& u = it->first;
    belief_type& blf = (*it).second;
    if (u->isVariable())
    {
      factor_value_type sum = factor_value_type(0.0);
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
    else // u is a factor, divide blf[0] / blf[1] = sum ( product * factor ) / sum (product)
    {
      if (blf[1] < factor_value_type(1e-15))
        blf[0] = factor_value_type(0.0);
      else
        blf[0] /= blf[1];
      blf[1] = factor_value_type(1.0);
    }
  }
}

template <typename _F>
    const typename NaiveInference<_F>::belief_type& NaiveInference<_F>::belief(const vertex_type& v) const
{
  return beliefs_.get(v);
}

END_GRAPHICAL_MODEL_NAMESPACE;

#endif // NAIVE_INFERENCE_H
