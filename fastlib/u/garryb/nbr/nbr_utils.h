#ifndef NBR_UTILS_H
#define NBR_UTILS_H

#include "kdtree.h"

namespace nbr_utils {
  success_t Load(const char *fname, TempCacheArray<Vector> *cache_out,
      index_t vectors_per_block);

  template<typename PointInfo, typename Node, typename Param>
  success_t LoadKdTree(struct datanode *datanode,
      const Param& param,
      TempCacheArray<PointInfo> *point_infos_out,
      TempCacheArray<Vector> *points_out,
      TempCacheArray<Node> *nodes_out) {
    index_t vectors_per_block = fx_param_int(
        datanode, "vectors_per_block", 256);
    
    fx_timer_start(datanode, "read");
    nbr_utils::Load(fx_param_str_req(datanode, ""), points_out,
        vectors_per_block);
    fx_timer_stop(datanode, "read");
    
    PointInfo blank_info;
    point_infos_out->Init(blank_info,
        points_out->end_index(), vectors_per_block);
    
    fx_timer_start(datanode, "tree");
    KdTreeMidpointBuilder<PointInfo, Node, Param> builder;
    builder.InitBuild(datanode, &param, points_out,
      point_infos_out, nodes_out);
    fx_timer_stop(datanode, "tree");
  }
  
  template<typename GNP, typename Solver>
  void SerialDualTreeMain(datanode *datanode, const char *gnp_name) {
    typename GNP::Param param;
    param.Init(fx_submodule(datanode, gnp_name, gnp_name));

    TempCacheArray<typename GNP::Point> q_points;
    TempCacheArray<typename GNP::QPointInfo> q_point_infos;
    TempCacheArray<typename GNP::QNode> q_nodes;
    TempCacheArray<typename GNP::Point> r_points;
    TempCacheArray<typename GNP::RPointInfo> r_point_infos;
    TempCacheArray<typename GNP::RNode> r_nodes;
    TempCacheArray<typename GNP::QResult> q_results;

    nbr_utils::LoadKdTree(fx_param_str_req(datanode, "q"),
        param, &q_point_infos, &q_points, &q_nodes);
    nbr_utils::LoadKdTree(fx_param_str_req(datanode, "r"),
        param, &r_point_infos, &r_points, &r_nodes);

    typename GNP::QResult default_result;
    default_result.Init(param);
    q_results.Init();

    Solver solver;
    solver.Init(fx_submodule(datanode, "solver", "solver"), param,
        &q_points, &q_point_infos, &q_nodes,
        &r_points, &r_point_infos, &r_nodes, &q_results);
    solver.Begin();
  }
};

#endif

