def config_doit(sysentry, files, params):
  script = files["script"].single(Types.SCRIPT)
  outdir = sysentry.bin_dir("arch", "kernel", "compiler")
  indir = sysentry.sys.source_dir
  sysentry.command("%s --genfiles_dir=%s --source_dir=%s" % (
      sq(script), sq(outdir), sq(indir)))
  gen_headers = ["base/basic_types.h"]
  return [(Types.HEADER, sysentry.file(h, "arch", "kernel", "compiler"))
      for h in gen_headers]

customrule(
    name = "config_headers",
    dependencies =
        {"script": [sourcerule(Types.SCRIPT, "../script/config.py")],
         "sources": sourcerules(Types.ANY, lglob("config/*.c"))},
    doit_fn = config_doit)

librule(
    sources = ["common.c", "cc.cc", "ccmem.cc"],
    headers = ["cc.h", "ccmem.h", "common.h", "compiler.h",
               "compiler_impl.h", "test.h", "fortran.h",
               "debug.h", "scale.h", ":config_headers"])
