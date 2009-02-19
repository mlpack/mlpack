
#ifndef __GRAPH_H
#define __GRAPH_H
#include "fastlib/fastlib.h"
#include <queue>

class Graph {
  typedef GenMatrix<bool> AdjacentMatrix;
  AdjacentMatrix adjacent;
  Matrix weight;
 public:
  bool isEdge(index_t i, index_t j) const { return adjacent.get(i, j); }
  double getW(index_t i, index_t j) const { return weight.get(i, j); }

  bool& refEdge(index_t i, index_t j) { return adjacent.ref(i, j); }
  double& refW(index_t i, index_t j) { return weight.ref(i, j); }

  index_t n_nodes() const { return adjacent.n_rows(); }

  void Init(index_t n) {
    adjacent.Init(n, n);
    weight.Init(n, n);
  }

  void InitFromFile(const char* f, double threshold = 0);
  void ThresholdEdges(double threshold);
};

typedef ArrayList<index_t> Path;

// Need isEdge() & n_nodes() functions
template <class Graph>
void BreadthFirstSearch(index_t s, index_t t, const Graph& g, Path* p) {
  std::queue<index_t> q;
  GenVector<bool> visited;
  GenVector<index_t> previous;

  visited.Init(g.n_nodes());
  previous.Init(g.n_nodes());

  visited.SetAll(false);
  previous.SetAll(-1);

  q.push(t); visited[t] = true;
  while (!q.empty() && !visited[s]) {
    int v = q.front(); q.pop();
    for (index_t u = 0; u < g.n_nodes(); u++)
      if (g.isEdge(u, v) && !visited[u]) {
	q.push(u); 
	previous[u] = v;
	visited[u] = true;
      }
  }
  p->Init();
  if (!visited[s]) return;
  p->PushBackCopy(s);
  while (s != t) {
    s = previous[s];
    p->PushBackCopy(s);
  }
}

template <class Graph>
class MaxFlowAumentedGraph {
  const Graph* g,
 public:
  MaxFlowAumentedGraph()
};

// weight as capacity
template <class Graph>
void MaxFlow(index_t s, index_t t, const Graph& g, Matrix* f) {
  MaxFlowAugmentedGraph<Graph> ag(g, f);
  
}

#endif
