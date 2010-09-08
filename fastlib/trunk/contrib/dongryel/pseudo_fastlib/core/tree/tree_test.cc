#include <armadillo>
#include "general_spacetree.h"
#include "gen_metric_tree.h"
#include "core/csv_parser/dataset_reader.h"
#include "core/metric_kernels/lmetric.h"
#include "core/table/table.h"

int main(int argc, char *argv[]) {

  int leaflen = 30;

  printf("Constructing the tree...\n");
  core::table::Table table;
  table.Init(std::string("test_data_3_1000.csv"));
  table.IndexData(leaflen);

  printf("Finished constructing the tree...\n");
  table.PrintTree();

  return 0;
}
