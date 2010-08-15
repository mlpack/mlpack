#ifndef FACTOR_GRAPH_H
#define FACTOR_GRAPH_H

#include <iostream>

#include "gm.h"

using namespace std;

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/**
  * A vertex could be variable or a factor
  * The data_ pointer could point to a Variable or a Factor
  */
class Vertex
{
  int type_;  // variable (0) or factor (1)
  void* data_;
public:
  Vertex(void* data, bool isFactor)
  {
    type_ = (isFactor) ? 1 : 0;
    data_ = data;
  }

  bool isFactor() const { return type_ == 1; }
  bool isVariable() const { return type_ == 0; }
  int type() const { return type_; }

  Variable* variable() const { DEBUG_ASSERT(type_ == 0); return (Variable*) data_; }
  void* factor() const { DEBUG_ASSERT(type_ == 1); return data_; }
};

/**
  * Implemetation of a factor graph, which consists of a set of factors
  * and variables that are connected as a graph
  * Parameters: <typename _F> must be a Factor type (see factor_template.h)
  */
template <typename _F>
class FactorGraph
{
public:
  typedef Vertex*                                                  vertex_type;
  typedef Vector<vertex_type>                                      vertex_vector_type;
  typedef Map<vertex_type, Vector<vertex_type> >                   neighbor_map_type;
  typedef _F                                                       factor_type;
  typedef typename _F::factor_value_type                           factor_value_type;
public:
  /** Add a factor to the graph, the graph is augmented with new nodes and edges */
  void add(const factor_type& f);

  /** Return the set of vertices in the graph */
  const vertex_vector_type& vertices() const { return vertices_; }

  /** Return the set of neighbors of a vertex */
  const vertex_vector_type& neighbors(const vertex_type& u) const
  {
    neighbor_map_type::const_iterator it = neighborMap_.find(u);
    DEBUG_ASSERT(it != neighborMap_.end());
    return (*it).second;
  }

  /** Get out the Factor in a factor vertex */
  const factor_type& factor(const vertex_type& u) const
  {
    DEBUG_ASSERT(neighborMap_.contains(u) && u->isFactor());
    return * ((factor_type*) u->factor());
  }

  /** Destructor, delete nodes and edges allocated by add() */
  ~FactorGraph();

  void print() const;
protected:
  vertex_vector_type vertices_;
  neighbor_map_type neighborMap_;
  Map<void*, vertex_type> vertexMap_;
};

// add new factor to the graph, allocate new nodes and edges (neighborMap_)
// if necessary
template <typename _F> void FactorGraph<_F>::add(const factor_type& f)
{
  factor_type* f_ = new factor_type(f);   // remember to delete (1)
  vertex_type vf_ = new Vertex(f_, true); // remember to delete (2)
  Vector<vertex_type> neighbor_;

  vertexMap_[(void*) f_] = vf_;
  const Domain& dom = f_->domain();
  BOOST_FOREACH(const Variable* var, dom)
  {
    vertex_type v;
    void* key = (void*) var;
    if (vertexMap_.contains(key))
      v = vertexMap_[key];
    else
    {
      v = new Vertex(key, false);         // remember to delete (2)
      vertexMap_[key] = v;
      vertices_ << v;
    }
    neighbor_ << v;
    neighborMap_[v] << vf_;
  }
  neighborMap_[vf_] = neighbor_;
  vertices_ << vf_;
}

template <typename _F> void FactorGraph<_F>::print() const
{
  BOOST_FOREACH(const vertex_type& u, vertices_)
  {
    const vertex_vector_type& nb = neighbors(u);

    cout << "Vertex: " << (u->isVariable() ? "Variable " : "Factor ");
    if (u->isVariable()) { u->variable()->print(); cout << endl; }
    else { cout << endl; factor(u).print(); }
    cout << "  Neighbors:";
    BOOST_FOREACH(const vertex_type& v, nb)
    {
      cout << " " << (v->type() == 0 ? v->variable()->name() + " (Variable)" : "(Factor)");
    }
    cout << endl;
  }
}

// Destructor, delete nodes and edgeds allocate by add()
template <typename _F> FactorGraph<_F>::~FactorGraph()
{
  for (unsigned int i = 0; i < vertices_.size(); i++)
  {
    cout << "DELETE Vertex " << (vertices_[i]->type() == 0 ?
                                 std::string("Variable ")+vertices_[i]->variable()->name() :
                                 std::string("Factor"))
         << endl;
    if (vertices_[i]->type() == 1) delete (factor_type*) vertices_[i]->factor();   //  (1)
    delete vertices_[i];                                                           //  (2)
  }
}

END_GRAPHICAL_MODEL_NAMESPACE;
#endif // FACTOR_GRAPH_H
