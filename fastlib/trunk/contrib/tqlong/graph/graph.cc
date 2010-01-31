#include "graph.h"

void Graph::InitFromFile(const char* f, double threshold) {
  data::Load(f, &weight);
  DEBUG_ASSERT(weight.n_rows() == weight.n_cols());
  la::TransposeSquare(&weight);
  adjacent.Init(weight.n_rows(), weight.n_rows());
  ThresholdEdges(threshold);
}

void Graph::ThresholdEdges(double threshold) {
  for (index_t i = 0; i < n_nodes(); i++)
    for (index_t j = 0; j < n_nodes(); j++)
      adjacent.ref(i, j) = weight.get(i, j) > threshold;
  //ot::Print(weight);
  //ot::Print(adjacent);
}

