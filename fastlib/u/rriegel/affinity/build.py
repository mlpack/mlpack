
librule(
  sources = ["affinity.cc"],
  deplibs = ["/linear:linear", "/tree:tree", "/base:base", "/xrun:xrun", "/file:file", "/collections:collections"])

binrule(
  name = "affinity_bin",
  linkables = [":affinity"])
