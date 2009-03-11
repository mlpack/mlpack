librule(
  sources = ["dataset.cc"],
  headers = ["dataset.h", "crossvalidation.h"],
  deplibs = ["fastlib/base:base", "fastlib/file:file", "fastlib/la:la"]
  )

binrule(
  name = "dataset_test",
  sources = ["dataset_test.cc"],
  deplibs = [":data", "fastlib/fx:fx", "fastlib/math:math"])

