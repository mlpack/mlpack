
#include "gm.h"
#include <fastlib/fastlib.h>

FactorGraph constructExampleFactorGraph() {
  String n0syms[] = {"rain", "sunny", "cloud"};
  String n1syms[] = {"hot", "cold"};
  String n2syms[] = {"0", "1"};
  String n3syms[] = {"a", "b", "c", "d"};

  Node nodes[4];
  nodes[0].Init(n0syms, 3);
  nodes[1].Init(n1syms, 2);  
  nodes[2].Init(n2syms, 2);
  nodes[3].Init(n3syms, 4);

  ArrayList<Node> nodes_list;
  nodes_list.InitCopy(nodes, 4);

  index_t f0ranges[] = {3, 2};
  index_t f1ranges[] = {2, 2};  
  index_t f2ranges[] = {2, 4};

  double f0vals[] = {1,2,3,4,5,6};
  double f1vals[] = {2,4,6,8};
  double f2vals[] = {3,6,9,12,15,18,21,24};

  Factor factors[3];
  factors[0].Init(f0ranges, 2, f0vals);
  factors[1].Init(f1ranges, 2, f1vals);
  factors[2].Init(f2ranges, 2, f2vals);

  ArrayList<Factor> factors_list;
  factors_list.InitCopy(factors, 3);

  index_t f0nodes[] = {0, 1};
  index_t f1nodes[] = {1, 2};
  index_t f2nodes[] = {1, 3};

  ArrayList<index_t> factor2node[3];
  factor2node[0].InitCopy(f0nodes, 2);
  factor2node[1].InitCopy(f1nodes, 2);
  factor2node[2].InitCopy(f2nodes, 2);

  ArrayList<ArrayList<index_t> > factor2node_list;
  factor2node_list.InitCopy(factor2node,3);

  FactorGraph fg;
  fg.Init(nodes_list, factors_list, factor2node_list);

  return fg;
}

int main() {
  FactorGraph fg = constructExampleFactorGraph();
  //ot::Print(fg, "factor graph", stdout);

  String syms[] = {"sunny", "cold", "1", "d"};
  ArrayList<String> syms_list;
  syms_list.InitCopy(syms, 4);

  fg.SetObserveds(syms_list);
  ot::Print(fg.jointProduct(), "joint product", stdout);
  return 0;
}

