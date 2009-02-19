#include "graph.h"

const fx_entry_doc maxflow_main_entries[] = {
  {"gcapa", FX_REQUIRED, FX_STR, NULL,
   "  File consists of the edges' capacity.\n"},
  {"gflow", FX_REQUIRED, FX_STR, NULL,
   "  Output file for the edges' flow.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc maxflow_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc maxflow_main_doc = {
  maxflow_main_entries, maxflow_main_submodules,
  "This is a program computing the maximum flow in a network.\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &maxflow_main_doc );
  const char *fn = fx_param_str_req(fx_root, "gcapa");
  const char *fout = fx_param_str_req(fx_root, "gflow");
  
  Graph g;
  g.InitFromFile(fn);

  //Path p;
  //BreadthFirstSearch(0, 3, g, &p);
  //ot::Print(p);
  
  Matrix f;
  f.Init(g.n_nodes(), g.n_nodes());
  f.SetZero();
  MaxFlow(0, 3, g, g.getW(), &f);
  la::TransposeSquare(&f);
  
  data::Save(fout, f);
  //ot::Print(f);

  fx_done(fx_root);
}

