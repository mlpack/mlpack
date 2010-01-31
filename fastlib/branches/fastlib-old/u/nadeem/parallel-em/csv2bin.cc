#include <fastlib/fastlib.h>
#include "fl_data_io.h"

int 
main(int argc, char* argv[])
{
  Dataset data;
  const char* srcname;
  const char* dstname;

  fx_init(argc, argv);

  // Get the source file name
  if (!(srcname = fx_param_str_req(NULL, "src"))) {
    fprintf(stderr, "Error in reading Source File\n");
    exit(1);
  }

  // Get the target file name
  if (!(dstname = fx_param_str_req(NULL, "dest"))) {
    fprintf(stderr, "Error in reading Destination File\n");
    exit(1);
  }

  // Get the config file name  
  if (!PASSED(data.InitFromFile(srcname))) {
    fprintf(stderr, "Couldn't open file '%s'.\n", srcname);
    exit(1);
  }

  write_matrix2bin(dstname, data.matrix());

  return 0;
}
