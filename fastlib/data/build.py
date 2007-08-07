librule(
  sources = ["dataset.cc"],
  headers = ["dataset.h", "crossvalidation.h"],
  deplibs = ["base:base", "file:file", "la:la"]
  )

binrule(
  name = "dataset_test",
  sources = ["dataset_test.cc"],
  deplibs = [":data", "fx:fx", "math:math"])

