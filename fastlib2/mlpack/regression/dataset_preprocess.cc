#include "fastlib/fastlib.h"

void FindIndexWithPrefix(Dataset &dataset, char *prefix,
			 ArrayList<int> &remove_indices) {

  // Get the dataset information containing the feature types and
  // names.
  DatasetInfo &info = dataset.info();
  ArrayList<DatasetFeature> &features = info.features();

  for(index_t i = 0; i < features.size(); i++) {

    // If a feature name with the desired prefix has been found, then
    // make sure it hasn't been selected before. If so, then add to
    // the remove indices.
    const String &feature_name = features[i].name();
    if(!strncmp(prefix, feature_name.c_str(), strlen(prefix))) {

      bool does_not_exist_yet = true;
      for(index_t j = 0; j < remove_indices.size(); j++) {
	if(remove_indices[j] == i) {
	  does_not_exist_yet = false;
	  break;
	}
      }
      if(does_not_exist_yet) {
	remove_indices.PushBackCopy(i);
      }
    }    
  }  
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv, NULL);

  // Read in the dataset from the file.
  Dataset initial_dataset;
  const char *dataset_name = fx_param_str_req(fx_root, "data");
  if(initial_dataset.InitFromFile(dataset_name) != SUCCESS_PASS) {
    FATAL("Could ont read the dataset %s", dataset_name);
  }

  // Now examine each feature name of the dataset, and construct the
  // indices.
  ArrayList<int> remove_indices;
  remove_indices.Init();
  char buffer[1000];
  do {
    printf("Input the prefix of the feature that you want to remove ");
    printf("(just press enter if you are done): ");
    fgets(buffer, 1000, stdin);

    if(strlen(buffer) == 0) {
      break;
    }
    FindIndexWithPrefix(initial_dataset, buffer, remove_indices);
  } while(true);

  fx_done(fx_root);
  return 0;
}
