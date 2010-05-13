
#include "gm.h"
#include <fastlib/fastlib.h>

void Factor::Init(const ArrayList<RangeType>& ranges) {
  index_t len = 1;
  
  for (int i = 0; i < ranges.size(); i++)
    len *= ranges[i];
  
  ranges_.InitCopy(ranges);
  vals_.Init(len);
  args_.Init(ranges_.size());
}

void Factor::Init(const ArrayList<RangeType>& ranges, const Vector& vals) {
  index_t len = 1;
  
  for (int i = 0; i < ranges.size(); i++)
    len *= ranges[i];
  
  DEBUG_ASSERT(len == vals.length());
  
  ranges_.InitCopy(ranges);
  vals_.Copy(vals);
  args_.Init(ranges_.size());
}

void Factor::Init(const RangeType* ranges, index_t n_nodes, 
		 const double* vals) {
  index_t len = 1;
  
  for (int i = 0; i < n_nodes; i++)
    len *= ranges[i];
  
  ranges_.InitCopy(ranges, n_nodes);
  vals_.Copy(vals, len);
  args_.Init(ranges_.size());
}

index_t Factor::GetIndex(const ArrayList<RangeType>& pos) {
  DEBUG_ASSERT(pos.size() == ranges_.size());
  
  index_t n = pos.size()-1;
  index_t rpos = pos[n];
  DEBUG_ASSERT(pos[n] < ranges_[n] && pos[n] >= 0);
  for (int i = n-1; i >= 0; i--) {
    DEBUG_ASSERT(pos[i] < ranges_[i] && pos[i] >= 0);
    rpos = rpos*ranges_[i]+pos[i];
  }
  
  return rpos;
}

void SumProductPassingAlgorithm::InitMessages(FactorGraphType& fg) {
  BGraphType& bgraph = fg.GetBGraph();
  
  // prepare spaces for messages between factor and node
  for (index_t i_factor = 0; i_factor < bgraph.n_factors(); i_factor++)
    for (index_t i_edge = 0; i_edge < bgraph.n_factornodes(i_factor); i_edge++)
      bgraph.msg_factor2node(i_factor, i_edge).Init
	(fg.GetFactor(i_factor).GetRange(i_edge));
  
  for (index_t i_node = 0; i_node < bgraph.n_nodes(); i_node++) {
    for (index_t i_edge = 0; i_edge < bgraph.n_nodefactors(i_node); i_edge++)
      bgraph.msg_node2factor(i_node, i_edge).Init
	(fg.GetNode(i_node).GetRange());
  }    
}

void SumProductPassingAlgorithm
  ::PassMessageNode2Factor(FactorGraphType& fg, index_t i_node, 
			   index_t i_edge) {
  BGraphType& bgraph = fg.GetBGraph();
  for (index_t val = 0; val < fg.GetNode(i_node).GetRange(); val++) {
    double s = 1;
    for (index_t i = 0; i < bgraph.n_nodefactors(i_node); i++)
      if (i != i_edge) {
	index_t i_factor = bgraph.factor(i_node, i);
	index_t c_edge = bgraph.factor_cedge(i_node, i);
	s *= bgraph.msg_factor2node(i_factor, c_edge)[val];
      }
    bgraph.msg_node2factor(i_node, i_edge)[val] = s;
    //printf("  val = %d s = %f\n", val, s);
  }
}

void SumProductPassingAlgorithm
  ::PassMessageFactor2Node(FactorGraphType& fg, index_t i_factor, 
			   index_t i_edge) {
  BGraphType& bgraph = fg.GetBGraph();
  index_t i_node = bgraph.node(i_factor, i_edge);
  for (index_t val = 0; val < fg.GetNode(i_node).GetRange(); val++) {
    fg.GetFactor(i_factor).SetArg(i_edge, val);
    double s = 0;
    VisitFactorArg(fg, i_factor, i_edge, 0, 1, s);
    bgraph.msg_factor2node(i_factor, i_edge)[val] = s;
    //printf("  val = %d s = %f\n", val, s);
  }
}

void SumProductPassingAlgorithm
  ::VisitFactorArg(FactorGraphType& fg, index_t i_factor, index_t i_edge, 
		   index_t i, double term, double& sum) {
  BGraphType& bgraph = fg.GetBGraph();
  if (i >= fg.GetFactor(i_factor).n_args()) {
    sum += term*fg.GetFactorVal(i_factor);
    return;
  }
  if (i_edge != i) {
    index_t i_node = bgraph.node(i_factor, i);
    index_t c_edge = bgraph.node_cedge(i_factor, i);
    for (index_t val = 0; val < fg.GetNode(i_node).GetRange(); val++) {
      fg.GetFactor(i_factor).SetArg(i, val);
      VisitFactorArg(fg, i_factor, i_edge, i+1, 
		     term*bgraph.msg_node2factor(i_node, c_edge)[val], sum);
    }
  }
  else
    VisitFactorArg(fg, i_factor, i_edge, i+1, term, sum);
}

void SumProductPassingAlgorithm
  ::PassMessages(FactorGraphType& fg, bool reverse) {
  BGraphType& bgraph = fg.GetBGraph();
  
  if (reverse) {
    for (index_t i = 0; i < n_orderedges(); i++) {
      index_t x = GetOrderEdgeFirst(i);
      index_t y = GetOrderEdgeSecond(i);
      //printf("x = %d y = %d\n", x, y);
      if (x < bgraph.n_nodes())
	PassMessageNode2Factor(fg, x, y);
      else 
	PassMessageFactor2Node(fg, x-bgraph.n_nodes(), y);
    }
  }
  else {
    for (index_t i = n_orderedges()-1; i >= 0; i--) {
      index_t x = GetOrderEdgeFirst(i);
      index_t y = GetOrderEdgeSecond(i);
      //printf("x = %d y = %d\n", x, y);
      if (x < bgraph.n_nodes()) {
	index_t i_factor = bgraph.factor(x, y);
	index_t c_edge = bgraph.factor_cedge(x, y);
	//printf("i_factor = %d c_edge = %d\n", i_factor, c_edge);
	PassMessageFactor2Node(fg, i_factor, c_edge);
      }
      else {
	index_t i_node = bgraph.node(x-bgraph.n_nodes(), y);
	index_t c_edge = bgraph.node_cedge(x-bgraph.n_nodes(), y);
	//printf("i_node = %d c_edge = %d\n", i_node, c_edge);
	PassMessageNode2Factor(fg, i_node, c_edge);
      }
    }
  }
}

double SumProductPassingAlgorithm
  ::NodeMarginalSum(FactorGraphType& fg, index_t i_node, 
		    ArrayList<double>* sum){
  BGraphType& bgraph = fg.GetBGraph();
  double Z = 0;
  sum->Init(fg.GetNode(i_node).GetRange());
  for (index_t val = 0; val < fg.GetNode(i_node).GetRange(); val++) {
    double s = 1;
    for (index_t i_edge = 0; 
	 i_edge < bgraph.n_nodefactors(i_node); i_edge++){
      index_t i_factor = bgraph.factor(i_node, i_edge);
      index_t c_edge = bgraph.factor_cedge(i_node, i_edge);
      s *= bgraph.msg_factor2node(i_factor, c_edge)[val];
    }
    (*sum)[val] = s;
    Z += s;
  }
  fg.SetZ(Z);
  return Z;
} 

