#ifndef NBR_UTILS_H
#define NBR_UTILS_H

namespace nbr_utils {
  success_t Load(const char *fname, TempCacheArray<Vector> *cache_out);

  template<typename PointInfo, typename Node, typename Param>
  success_t LoadKdTree(struct datanode *datanode,
      const Param& param,
      CacheArray<PointInfo> *point_infos_to_rearrange,
      TempCacheArray<Vector> *points_out,
      TempCacheArray<Node> *nodes_out) {
    nbr_utils::Load(fx_param_str_req(datanode, ""), points_out);
    KdTreeMidpointBuilder<TPointInfo, TNode, TParam> builder;
    builder.InitBuild(datanode, &param, points_out,
      point_infos_to_rearrange, nodes_out);
  }
};

#endif

