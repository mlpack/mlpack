
binrule(
   name = "nnsvm_bin",
   sources = ["nnsvm.cc"],
   headers = ["nnsmo.h", "nnsvm.h"],
   linkables = ["fastlib:fastlib"])

