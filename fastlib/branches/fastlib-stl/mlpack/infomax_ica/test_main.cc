/**
 * @file main.cc
 *
 * Test driver for our infomax ICA method.
 */

#include "infomax_ica.h"
#include "test_infomax_ica.h"
#include "fastlib/fastlib.h"

const fx_entry_doc infomax_ica_main_entries[] = {
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc infomax_ica_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc infomax_ica_main_doc = {
  infomax_ica_main_entries, infomax_ica_main_submodules,
  "This performs ICA decomposition on a given dataset using the Infomax method.\n"
};

int main(int argc, char *argv[]) {
  fx_module *root = fx_init(argc, argv, &infomax_ica_main_doc);

  TestInfomaxICA *testica = new TestInfomaxICA();
  testica->Init();
  testica->TestAll();

  fx_done(NULL);
}
