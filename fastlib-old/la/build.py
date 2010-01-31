# Scroll down to see the main part.
# (This first part is dedicated to compiling and installing LAPACK.)

wgetrule(
    name = "blaspack_tgz",
    type = Types.ANY,
    url = "http://www.cc.gatech.edu/~garryb/fastlib/blaspack.tgz")

wgetrule(
    name = "complex_blaspack_tgz",
    type = Types.ANY,
    url = "http://www.cc.gatech.edu/~gtg739c/complex_blaspack.tgz")

def gen_compile_complex_lapack(sysentry, files, params):
  blaspack_tgz = files["complex_blaspack_tgz"].single(Types.ANY)
  libblaspack = sysentry.file("KEEP/libcomplexblaspack.a", "arch", "kernel", "compiler")
  workspace_dir = os.path.join(os.path.dirname(libblaspack.name), "libcomplexblaspack_workspace")
  compiler_info = compilers[params["compiler"]]
  compiler = compiler_info.compiler_program("f")
  # Make sure we won't rm -rf anything bad
  assert "libcomplexblaspack_workspace" in workspace_dir
  sysentry.command("echo '... Extracting FORTRAN files...'")
  sysentry.command("mkdir -p %s" % sq(workspace_dir))
  sysentry.command("cd %s && tar -xzf %s" % (sq(workspace_dir), sq(blaspack_tgz.name)))
  sysentry.command("echo '*** Compiling COMPLEX LAPACK with BLAS reference implementation.'")
  sysentry.command("echo '!!! LAPACK WARNING: For better performance, install ATLAS or Intel MKL.'")
  sysentry.command("echo '... Our compilation differs slightly from regular LAPACK/BLAS:'")
  sysentry.command("echo '... NOTE 2: We omit real-number routines (halves compile time).'")
  sysentry.command("echo '... NOTE 3: We require case sensitivity for LAPACK/BLAS string parameters.'")
  sysentry.command("echo '... This may take several minutes (about 800 FORTRAN files).'")
  # Loop unrolling is not useful on modern architectures (and bloats EXE size).
  # Thus, we only compile -O2.
  sysentry.command("cd %s && %s -O2 -c src/*.f" % (sq(workspace_dir), sq(compiler)))
  sysentry.command("echo '... Almost done with LAPACK/BLAS...'")
  #noopt = "zlamch clamch" # these have to be compiled without optimization
  #noopt_real = " ".join(sq(["src/%s.f" % x for x in noopt.split()]))
  #sysentry.command("cd %s && %s -O0 -c %s" % (sq(workspace_dir), sq(compiler), noopt_real))
  sysentry.command("cd %s && ar r %s *.o" % (sq(workspace_dir), sq(libblaspack.name)))
  sysentry.command("echo '... Created archive, cleaning up.'")
  sysentry.command("rm -rf %s" % sq(workspace_dir))
  sysentry.command("echo '*** Done with LAPACK and BLAS!'")
  return [(Types.LINKABLE, libblaspack)]


def gen_compile_lapack(sysentry, files, params):
  blaspack_tgz = files["blaspack_tgz"].single(Types.ANY)
  libblaspack = sysentry.file("KEEP/libblaspack.a", "arch", "kernel", "compiler")
  workspace_dir = os.path.join(os.path.dirname(libblaspack.name), "libblaspack_workspace")
  compiler_info = compilers[params["compiler"]]
  compiler = compiler_info.compiler_program("f")
  # Make sure we won't rm -rf anything bad
  assert "libblaspack_workspace" in workspace_dir
  sysentry.command("echo '... Extracting FORTRAN files...'")
  sysentry.command("mkdir -p %s" % sq(workspace_dir))
  sysentry.command("cd %s && tar -xzf %s" % (sq(workspace_dir), sq(blaspack_tgz.name)))
  sysentry.command("echo '*** Compiling LAPACK with BLAS reference implementation.'")
  sysentry.command("echo '!!! LAPACK WARNING: For better performance, install ATLAS or Intel MKL.'")
  sysentry.command("echo '... Our compilation differs slightly from regular LAPACK/BLAS:'")
  sysentry.command("echo '... NOTE 2: We omit complex-number routines (halves compile time).'")
  sysentry.command("echo '... NOTE 3: We require case sensitivity for LAPACK/BLAS string parameters.'")
  sysentry.command("echo '... This may take several minutes (about 800 FORTRAN files).'")
  # Loop unrolling is not useful on modern architectures (and bloats EXE size).
  # Thus, we only compile -O2.
  sysentry.command("cd %s && %s -O2 -c src/*.f" % (sq(workspace_dir), sq(compiler)))
  sysentry.command("echo '... Almost done with LAPACK/BLAS...'")
  noopt = "dlamch slamch" # these have to be compiled without optimization
  noopt_real = " ".join(sq(["src/%s.f" % x for x in noopt.split()]))
  sysentry.command("cd %s && %s -O0 -c %s" % (sq(workspace_dir), sq(compiler), noopt_real))
  sysentry.command("cd %s && ar r %s *.o" % (sq(workspace_dir), sq(libblaspack.name)))
  sysentry.command("echo '... Created archive, cleaning up.'")
  sysentry.command("rm -rf %s" % sq(workspace_dir))
  sysentry.command("echo '*** Done with LAPACK and BLAS!'")
  return [(Types.LINKABLE, libblaspack)]


customrule(
    name = "libcomplexblaspack",
    dependencies = {"complex_blaspack_tgz": [find(":complex_blaspack_tgz")]},
    doit_fn = gen_compile_complex_lapack)


customrule(
    name = "libblaspack",
    dependencies = {"blaspack_tgz": [find(":blaspack_tgz")]},
    doit_fn = gen_compile_lapack)

#---- This is the main part of LA package

librule(
    sources = ["uselapack.cc"],
    headers = ["matrix.h", "la.h", "uselapack.h", "clapack.h", "blas.h"],
    tests = ["uselapack_test.cc"],
    deplibs = ["base:base", "col:col", ":libblaspack", "math:math"])
