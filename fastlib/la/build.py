
class LaProtoRule(dep.Rule):
  def __init__(self):
    self.script = sourcerule(Types.SCRIPT, "proto.py")
    self.sources = [] # TODO: Include files this depends on
    dep.Rule.__init__(self, script=[self.script], sources=self.sources)
  def doit(self, sysentry, state, files, params):
    script = files["script"].single(Types.SCRIPT)
    proto_header = sysentry.file("la/proto.h", "arch", "kernel", "compiler")
    sysentry.command("python %s --outfile=%s" % (script.name, proto_header.name))
    gen_headers = ["la/proto.h"]
    return [(Types.HEADER, proto_header)]

librule(
    sources = [],
    headers = ["matrix.h", "la.h"],
    deplibs = ["base:base", "col:col"])

binrule(
    name = "lapack_test",
    sources = ["lapack.cc"],
    headers = [LaProtoRule()],
    linkables = [":la"])

binrule(
    name = "la_test",
    sources = ["la_test.cc"],
    headers = [],
    linkables = [":la"])
