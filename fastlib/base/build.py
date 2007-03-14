
class ConfigHeadersRule(dep.Rule):
  def __init__(self):
    self.script = sourcerule(Types.SCRIPT, "config.py")
    self.sources = sourcerules(Types.ANY, lglob("config/*.c"))
    dep.Rule.__init__(self, script=[self.script], sources=self.sources)
  def doit(self, sysentry, state, files, params):
    script = files["script"].single(Types.SCRIPT)
    outdir = sysentry.bin_dir("arch", "kernel", "compiler")
    indir = sysentry.sys.source_dir
    sysentry.command("%s --genfiles_dir=%s --source_dir=%s" % (script, outdir, indir))
    gen_headers = ["base/basic_types.h"]
    return [(Types.HEADER, sysentry.file(h, "arch", "kernel", "compiler"))
        for h in gen_headers]

register(name = "config_headers",
    rule = ConfigHeadersRule())

librule(
    sources = ["common.c", "cc.cc", "ccmem.cc"],
    headers = ["cc.h", "ccmem.h", "common.h", "compiler.h",
               "compiler_impl.h",
               "debug.h", "scale.h", ":config_headers"])

