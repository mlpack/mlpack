/** @file dualtree_load_balancer.h
 *
 *  @author Dongryeol Lee (dongryel@.cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DUALTREE_LOAD_BALANCER_H
#define CORE_PARALLEL_DUALTREE_LOAD_BALANCER_H

#include <boost/mpi.hpp>
#include <parmetis/parmetislib.h>

namespace core {
namespace parallel {
class DualtreeLoadBalancer {
  public:

    template<typename TreeType>
    static void Compute(
      boost::mpi::communicator &comm,
      const std::vector<TreeType *> &local_query_subtrees,
      const std::vector <
      std::vector <
      std::pair<int, int> > > &essential_reference_subtrees_to_send,
      const std::vector <
      std::vector <
      std::pair<int, int> > > &reference_subtrees_to_receive,
      int num_reference_subtrees_to_receive) {

      // Figure out the subtree distribution by computing the prefix
      // sum. The number of vertices on each process is one more than
      // the number of subtrees.
      boost::scoped_array<int> query_subtree_distribution(
        new int[ comm.size() + 1 ]);
      int num_local_vertices = local_query_subtrees.size() + 1 ;
      query_subtree_distribution[comm.rank() + 1] =
        boost::mpi::scan(comm, num_local_vertices, std::plus<int>());
      boost::mpi::all_gather(
        comm, query_subtree_distribution[comm.rank() + 1],
        ((int *) query_subtree_distribution.get() + 1));
      query_subtree_distribution[ 0 ] = 0;

      // Fill out the local CSR information here. For each vertex
      // representing a query subtree, it is not adjacent to another
      // query subtree. For each vertex representing a MPI process, it
      // is adjacent to every MPI process with the negative infinity
      // weight to encourage a partition with each vertex belonging to a
      // different one.
      std::vector<int> local_adjacency;
      boost::scoped_array<int> local_xadjacency(
        new int[ num_local_vertices + 1 ]);
      for(int i = 0; i <= num_local_vertices; i++) {
        local_xadjacency[i] = 0;
      }
      boost::scoped_array<int> local_vertex_weights(
        new int[num_local_vertices]);
      std::vector<int> local_edge_weights;

      // First, fill out the edges going out from the process vertex.
      local_vertex_weights[0] = 0;
      for(int i = 0;
          i < static_cast<int>(essential_reference_subtrees_to_send.size());
          i++) {
        const std::vector< std::pair<int, int> > &send_list =
          essential_reference_subtrees_to_send[ i ];

        if(i != comm.rank()) {

          // Otherwise, the weight is a very small positive integer.
          local_edge_weights.push_back(1);

          // Have the process vertex point to the current process vertex.
          local_adjacency.push_back(query_subtree_distribution[i]);
        }
        for(unsigned int j = 0; j < send_list.size(); j++) {
          int query_subtree_vertex_id =
            query_subtree_distribution[i] + j + 1;

          // The edge weight from the process vertex to the query
          // subtree is the number of outgoig reference
          // trees. Multiply the edge weight by 10.
          local_edge_weights.push_back(send_list.size() * 10);
          local_adjacency.push_back(query_subtree_vertex_id);
        }
      }
      local_xadjacency[1] = local_adjacency.size();

      // Up to here, we are done with the adjacency list of the
      // process vertex.

      // Now fill out the vertex weights for the query subtrees, which
      // is proportional to the number of reference subtrees to
      // compute on.
      for(int i = 1; i < num_local_vertices; i++) {
        local_vertex_weights[i] = num_reference_subtrees_to_receive;
      }

      // Now outgoing edges for the subtree vertices.
      for(int i = 1; i < num_local_vertices; i++) {
        for(int j = 0; j < comm.size(); j++) {
          local_adjacency.push_back(
            query_subtree_distribution[ j ]);
          local_edge_weights.push_back(
            reference_subtrees_to_receive[j].size());
        }
        local_xadjacency[i + 1] = local_xadjacency[i] + comm.size();
      }


      printf("Checking on %d with %d subtrees.\n", comm.rank(),
             local_query_subtrees.size());
      printf("Xadjacency: ");
      for(int i = 0; i <= num_local_vertices; i++) {
        printf("%d ", local_xadjacency[i]);
      }
      printf("\n");
      printf("Adjacency: ");
      for(unsigned int i = 0; i < local_adjacency.size(); i++) {
        printf("%d ", local_adjacency[i]);
      }
      printf("\nEdge weights: ");
      for(unsigned int i = 0; i < local_edge_weights.size(); i++) {
        printf("%d ", local_edge_weights[i]);
      }
      printf("\n\n");

      // Parameters necessary for calling the parallel graph
      // partitioner.
      int wgtflag = 3;
      int numflag = 0;
      int ncon = 1;
      int nparts = comm.size();
      boost::scoped_array<float> tpwgts(new float[ comm.size()]);
      for(int i = 0; i < comm.size(); i++) {
        tpwgts[i] = 1.0 / static_cast<float>(comm.size());
      }
      float ubvec = 1.05;
      int options = 0;
      int edgecut;
      boost::scoped_array<int> part(new int[ num_local_vertices ]);
      int *adjncy = &(local_adjacency[0]);
      int *adjwgt = &(local_edge_weights[0]);

      MPI_Comm mpi_comm = comm.operator MPI_Comm();
      ParMETIS_V3_PartKway(
        query_subtree_distribution.get(),
        local_xadjacency.get(),
        adjncy,
        local_vertex_weights.get(),
        adjwgt,
        &wgtflag,
        &numflag,
        &ncon,
        &nparts,
        tpwgts.get(),
        &ubvec,
        &options,
        &edgecut,
        part.get(),
        &mpi_comm);
    }
};
}
}

#endif
