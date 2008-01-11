
binrule(
   name = "svm_bin",
   sources = ["svm.cc"],
   headers = ["smo.h", "svm.h"],
   linkables = ["fastlib:fastlib"])

