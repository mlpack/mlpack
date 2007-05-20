
template<typename PointInfo, typename Node, typename Param>
class LocalTreeManager {
 private:
  SmallCache point_cache;
  SmallCache point_info_cache;
  SmallCache node_cache;

 public:
  TreeManager();
  ~TreeManager();
  
  void BuildTree(struct datanode *datanode,
      const Param& param,
      CacheArray<Vector> *points_out,
      CacheArray<PointInfo> *point_infos_out,
      CacheArray<Node> *nodes_out) {
    const char *fname = fx_param_str_req(datanode, "");
    Matrix matrix;
    data::Load(fname, &matrix);
    
    Vector first_data;
    matrix.MakeColumnVector(0, &first_data);
    
    PointInfo blank_info; // WALDO
    
    BlockActionHandler *point_handler =
        new CacheArrayBlockActionHandler<Vector>(first_data);
    
    KdTreeMidpointBuilder<PointInfo, Node, Param> tree_builder;
    tree_builder.InitBuild(
        fx_submodule(datanode, "tree", "tree"),
        &param,
        points_out,
        point_infos_out,
        nodes_out);
  }
}


outline

template<typename GNP, typename Solverk>
class DualTreeRunner {
 private:
  LocalTreeManager q_;
  LocalTreeManager r_;
  SmallCache WALDO;
  
 public:
  void InitRun(struct datanode *datanode) {
    param_.Init(fx_submodule(datanode, "param"));
    
    
  }
};

int main(void) {
  
}


