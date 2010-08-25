#ifndef __MESSAGE_PENDING_INFERENCE_H
#define __MESSAGE_PENDING_INFERENCE_H

#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** The sum-product algorithm with message priority queue:
  * + Messages are picked to update according to theirs changes in previous iteration.
  * + Messages are propagated along all edges for a certain number of iterations
  *   while the total change of messages is bigger than change tolerance.
  * + Messages are used to calculate beliefs of variable-vertex and average of factor-vertex.
  */
template <typename _F>
    class MessagePendingInference : public MessagePriorityInference<_F>
{
public:
  typedef MessagePriorityInference<_F>                       _Base;
  typedef typename _Base::factor_type                        factor_type;         // map from Assignment --> factor_value_type
  typedef typename _Base::factor_value_type                  factor_value_type;   // a numeric type
  typedef typename _Base::graph_type                         graph_type;          // FactorGraph<factor_type>
  typedef typename _Base::vertex_type                        vertex_type;         // Vertex* (see factor_graph.h)
  typedef typename _Base::vertex_vector_type                 vertex_vector_type;  // Vector<vertex_type>
  typedef typename _Base::belief_type                        belief_type;         // Map<Value, factor_value_type>
  typedef typename _Base::belief_map_type                    belief_map_type;     // Map<vertex_type, belief_type>
  typedef typename _Base::message_type                       message_type;        // Map<Value, factor_value_type>
  typedef typename _Base::messages_map_type                  messages_map_type;   // Map<vertex_type, Map<vertex_type, message_type> >
                                                                                  // usage: message[u][v] where u,v are vertex_type
public:
  /** Preparing inference on a graph */
  MessagePendingInference(const graph_type& graph, ConvergenceMeasure cvm = ConvergenceMeasure())
    : _Base(graph, cvm) {}

  /** The inference algorithm */
  void run();
protected:

};

template <typename _F> void MessagePendingInference<_F>::run()
{
  cout << "----------------------- Message Pending Sum-Product Inference -----------------" << endl;
  this->initBeliefs();
  this->initMessages();
  cout << "---------------------- iter = " << (this->curCvm_.iter_=0) << " Initializiing ----------------" << endl;
  this->calculateBeliefs();
  this->normalizeBeliefs();
}

END_GRAPHICAL_MODEL_NAMESPACE;

#endif
