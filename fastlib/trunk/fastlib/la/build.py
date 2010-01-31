# Scroll down to see the main part.
# (This first part is dedicated to compiling and installing LAPACK.)
import commands;
import sys

wgetrule(
    name = "blaspack_tgz",
    type = Types.ANY,
    url = "http://www.cc.gatech.edu/~garryb/fastlib/blaspack.tgz")

wgetrule(
    name = "complex_blaspack_tgz",
    type = Types.ANY,
    url = "http://www.cc.gatech.edu/~gtg739c/complex_blaspack.tgz")

wgetrule(
    name = "blas_lock",
    type = Types.ANY,
    url = "www.gatech.edu/blas.lock")

wgetrule(
    name = "lapack_lock",
    type = Types.ANY,
    url = "www.gatech.edu/lapack.lock")


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
  # Thus, we only compile -O.
  sysentry.command("cd %s && %s -O -c src/*.f" % (sq(workspace_dir), sq(compiler)))
  sysentry.command("echo '... Almost done with LAPACK/BLAS...'")
  #noopt = "zlamch clamch" # these have to be compiled without optimization
  #noopt_real = " ".join(sq(["src/%s.f" % x for x in noopt.split()]))
  #sysentry.command("cd %s && %s -O0 -c %s" % (sq(workspace_dir), sq(compiler), noopt_real))
  sysentry.command("cd %s && ar r %s *.o" % (sq(workspace_dir), sq(libblaspack.name)))
  sysentry.command("echo '... Created archive, cleaning up.'")
  sysentry.command("rm -rf %s" % sq(workspace_dir))
  # Export it if the user requested.
  if params["prefix"] != "":
    destination_parent_dir = params["prefix"].rstrip("/").rstrip(os.sep) + "/lib/"
    sysentry.command("mkdir -p " + destination_parent_dir)
    sysentry.command("cp -f " + (sq(libblaspack.name)) + " " + destination_parent_dir)
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
  # Thus, we only compile -O.
  sysentry.command("cd %s && %s -O -c src/*.f" % (sq(workspace_dir), sq(compiler)))
  sysentry.command("echo '... Almost done with LAPACK/BLAS...'")
  noopt = "dlamch slamch" # these have to be compiled without optimization
  noopt_real = " ".join(sq(["src/%s.f" % x for x in noopt.split()]))
  sysentry.command("cd %s && %s -O0 -c %s" % (sq(workspace_dir), sq(compiler), noopt_real))
  sysentry.command("cd %s && ar r %s *.o" % (sq(workspace_dir), sq(libblaspack.name)))
  sysentry.command("echo '... Created archive, cleaning up.'")
  sysentry.command("rm -rf %s" % sq(workspace_dir))
  if params["prefix"] != "":
    destination_parent_dir = params["prefix"].rstrip("/").rstrip(os.sep) + "/lib/"
    sysentry.command("mkdir -p " + destination_parent_dir)
    sysentry.command("cp -f " + (sq(libblaspack.name)) + " " + destination_parent_dir)
  sysentry.command("echo '*** Done with LAPACK and BLAS!'")
  return [(Types.LINKABLE, libblaspack)]

def make_blas(sysentry, files, params):
  libblas = sysentry.file("KEEP/libblas.a", "arch", "kernel", "compiler")
  blas_lock = files["blas_lock"].single(Types.ANY)
  if commands.getoutput("ls " + sq(blas_lock.name))==sq(blas_lock.name):
    return [(Types.LINKABLE, libblas)]
  else:
    print "First time installing BLAS, Here are your options:"
    print "1)Press 1 if you want fastlib to download and install it from Netlib website"
    print "2)Press 2 if you think you have a better version already installed press 2"
    print "  You will be asked to give the full path along with the library name (/path/libexample.a)"
    print " UBUNTU users should manualy install the atlas lapack/blas through the synaptic package manager"
    print " choose option 2, the library path is /usr/lib/atlas/libblas.a "
    print "  if after giving the path you realize that it is wrong or it doesn't work"
    print "  delete the $instalation_dir/fastlib/la/blas.lock file and build your code again"
    print "3)If you have cygwin choose this option, make sure you have installed lapack with the cygwin installer"
    resp=0;
    while resp!=1 and resp!=2 and resp!=3:
      resp=input("Give me your choice now: ");
      if resp!=1 and resp!=2 and resp!=3:
        print "You chose "+str(resp)+ " which is not a valid choice"
      commands.getoutput("rm -f " + sq(libblas.name));

    if resp==3:
      print commands.getoutput("ln -s /lib/libblas.dll.a "+ sq(libblas.name));
    else:
      if resp==2:
        blas_lib=input("Give me now the full path with the BLAS library name in quotes ie \"/usr/lib/libblas.a\": ");
        print "I am creating a symbolic link to "+ str(blas_lib)
        commands.getoutput("rm -f " + sq(libblas.name));
        print commands.getoutput("ln -s "+str(blas_lib)+ " "+ sq(libblas.name));
      else:
        if resp==1:
          print "Now I will download BLAS from netlib.org"
          workspace_dir = os.path.join(os.path.dirname(libblas.name), "netlib_workspace1")
        # Make sure we won't rm -rf anything bad
        assert "netlib_workspace1" in workspace_dir
        sysentry.command("echo '... Extracting  files...'")
        sysentry.command("mkdir -p %s" % sq(workspace_dir))
        if os.path.basename(params["downloader"]) == "wget":
          sysentry.command("cd %s && %s http://www.netlib.org/blas/blas.tgz" % ((sq(workspace_dir)), sq(params["downloader"])))
        else:
            sysentry.command("cd %s && %s http://www.netlib.org/blas/blas.tgz -o blas.tgz" % ((sq(workspace_dir)), sq(params["downloader"])))
        sysentry.command("cd %s && tar -xzf %s" % (sq(workspace_dir), sq("blas.tgz")))
        compiler_info = compilers[params["compiler"]]
        compiler = compiler_info.compiler_program("f")
        if compiler == "gfortran":
          sysentry.command("cd %s/BLAS && sed -i.backup -e 's/g77/gfortran/' make.inc" % (sq(workspace_dir)))
          if compiler == "g77":
            sysentry.command("cd %s/BLAS && sed -i.backup -e 's/gfortran/g77/' make.inc" % (sq(workspace_dir)))
        sysentry.command("echo '*** Compiling  BLAS from netlib.org'")
        sysentry.command("echo '!!! BLAS WARNING: For better performance, install ATLAS or Intel MKL.'")
        sysentry.command("echo '... This may take several minutes (about 800 FORTRAN files).'")
        sysentry.command("cd %s/BLAS && make all" % (sq(workspace_dir)))
        sysentry.command("echo '... Almost done with BLAS...'")
        sysentry.command("cd %s/BLAS && mv blas*.a %s" % (sq(workspace_dir), sq(libblas.name)))
        sysentry.command("echo '... Created archive, cleaning up.'")
        sysentry.command("rm -rf %s" % sq(workspace_dir))
        if params["prefix"] != "":
          destination_parent_dir = params["prefix"].rstrip("/").rstrip(os.sep) + "/lib/"
          sysentry.command("mkdir -p " + destination_parent_dir)
          sysentry.command("cp -f " + (sq(libblas.name)) + " " + destination_parent_dir)
          sysentry.command("echo '*** Done with  and BLAS!'")
  print "Generating a blas.lock file, if you want to reinstall differently delete it"
  print commands.getoutput("echo lock"  + " >"+sq(blas_lock.name));
  return [(Types.LINKABLE, libblas)]



def make_lapack(sysentry, files, params):
  liblapack = sysentry.file("KEEP/liblapack.a", "arch", "kernel", "compiler");
  lapack_lock = files["lapack_lock"].single(Types.ANY)
  if commands.getoutput("ls "+ sq(lapack_lock.name))==sq(lapack_lock.name):
    return [(Types.LINKABLE, liblapack)]
  else:
    print "First time installing LAPACK, Here are your options:"
    print "1)Press 1 if you want fastlib to download and install it from Netlib website"
    print "2)Press 2 if you think you have a better version already installed press 2"
    print " UBUNTU users should manualy install the atlas lapack/blas through the synaptic package manager"
    print " choose option 2, the library path is /usr/lib/atlas/liblapack.a "
    print "  You will be asked to give the full path along with the library name (/path/libexample.a)"
    print "  if after giving the path you realize that it is wrong or it doesn't work"
    print "  delete the $instalation_dir/fastlib/la/lapack.lock file and build your code again"
    print "3)If you have cygwin choose this option, make sure you have installed lapack with the cygwin installer"

    resp=0;
    while resp!=1 and resp!=2 and resp!=3:
      resp=input("Give me your choice now: ");
      if resp!=1 and resp!=2 and resp!=3:
        print "You chose "+str(resp)+ " which is not a valid choice"
      commands.getoutput("rm -f " + sq(liblapack.name));

    if resp==3:
      print commands.getoutput("ln -s /lib/liblapack.dll.a " + sq(liblapack.name));

    if resp==2:
      lapack_lib=input("Give me now the full path with the LAPACK library name in quotes ie \"/usr/lib/liblapack.a\": ");
      print "I am creating a symbolic link to "+ str(lapack_lib)
      commands.getoutput("rm -f " + sq(liblapack.name));
      print commands.getoutput("ln -s "+str(lapack_lib)+ " " + sq(liblapack.name));
    if resp==1:
      print "Now I will download LAPACK from netlib.org"
      workspace_dir = os.path.join(os.path.dirname(liblapack.name), "netlib_workspace2")
      compiler_info = compilers[params["compiler"]]
      compiler = compiler_info.compiler_program("f")
      # Make sure we won't rm -rf anything bad
      assert "netlib_workspace2" in workspace_dir
      sysentry.command("echo '... Extracting  files...'")
      sysentry.command("mkdir -p %s" % sq(workspace_dir))
      if os.path.basename(params["downloader"]) == "wget":
        sysentry.command("cd %s && %s http://www.netlib.org/lapack/lapack.tgz" % ((sq(workspace_dir)), params["downloader"]))
      else:
         sysentry.command("cd %s && %s http://www.netlib.org/lapack/lapack.tgz -o lapack.tgz" % ((sq(workspace_dir)), params["downloader"]))
      sysentry.command("cd %s && tar -xzf %s" % (sq(workspace_dir), sq("lapack.tgz")))
      sysentry.command("echo '*** Compiling  LAPACK  from netlib.org'")
      sysentry.command("echo '!!! LAPACK WARNING: For better performance, install ATLAS or Intel MKL.'")
      sysentry.command("echo '... This may take several minutes (about 800 FORTRAN files).'")
      sysentry.command("cd %s/lapack* && sed -i.backup -e 's/; .\/testlsame\;//' Makefile" % (sq(workspace_dir)))
      sysentry.command("cd %s/lapack* && sed -i.backup -e 's/.\/testslamch\;//' Makefile" % (sq(workspace_dir)))
      sysentry.command("cd %s/lapack* && sed -i.backup -e 's/.\/testdlamch\;//' Makefile" % (sq(workspace_dir)))
      sysentry.command("cd %s/lapack* && sed -i.backup -e 's/.\/testsecond\;//' Makefile" % (sq(workspace_dir)))
      sysentry.command("cd %s/lapack* && sed -i.backup -e 's/.\/testdsecnd\;//' Makefile" % (sq(workspace_dir)))
      sysentry.command("cd %s/lapack* && sed -i.backup -e 's/.\/testversion//' Makefile" % (sq(workspace_dir)))
      if compiler == "gfortran":
        sysentry.command("cd %s/lapack* && sed -i.backup -e 's/g77/gfortran/' make.inc.example" % (sq(workspace_dir)))
      if compiler == "g77":
        sysentry.command("cd %s/lapack* && sed -i.backup -e 's/gfortran/g77/' make.inc.example" % (sq(workspace_dir)))
      # This is fixing Bill's problem.
      sysentry.command("cd %s/lapack* && echo 'TIMER = INT_CPU_TIME' >> make.inc.example" % (sq(workspace_dir)))
      # This is fixing the compilation flag to be something other than -g option.
      sysentry.command("cd %s/lapack* && echo 'FORTRAN = gfortran -fimplicit-none -O' >> make.inc.example" % (sq(workspace_dir)))
      sysentry.command("cd %s/lapack* && echo 'LOADER = gfortran -O' >> make.inc.example" % (sq(workspace_dir)))
      sysentry.command("cd %s/lapack* && mv make.inc.example make.inc" % (sq(workspace_dir)) )
      sysentry.command("cd %s/lapack* && make lapacklib" % (sq(workspace_dir)))
      sysentry.command("echo '... Almost done with LAPACK...'")
      sysentry.command("cd %s/lapack* && mv lapack*.a %s" % (sq(workspace_dir), sq(liblapack.name)))
      sysentry.command("echo '... Created archive, cleaning up.'")
      sysentry.command("rm -rf %s" % sq(workspace_dir))
      if params["prefix"] != "":
        destination_parent_dir = params["prefix"].rstrip("/").rstrip(os.sep) + "/lib/"
        sysentry.command("mkdir -p " + destination_parent_dir)
        sysentry.command("cp -f " + (sq(liblapack.name)) + " " + destination_parent_dir)
      sysentry.command("echo '*** Done with LAPACK !'")
  print "Generating a lapack.lock file, if you want to reinstall differently delete it"
  print commands.getoutput("echo lock"  + " > " + sq(lapack_lock.name));
  return [(Types.LINKABLE, liblapack)]



customrule(
    name = "libcomplexblaspack",
    dependencies = {"complex_blaspack_tgz": [find(":complex_blaspack_tgz")]},
    doit_fn = gen_compile_complex_lapack)


customrule(
    name = "libblaspack",
    dependencies = {"blaspack_tgz": [find(":blaspack_tgz")]},
    doit_fn = gen_compile_lapack)

customrule(
    name="netlib_blas",
    dependencies = {"blas_lock": [find(":blas_lock")]},
    doit_fn = make_blas
    )

customrule(
    name="netlib_lapack",
    dependencies = {"lapack_lock": [find(":lapack_lock")]},
    doit_fn = make_lapack
    )

#---- This is the main part of LA package

librule(
    sources = ["uselapack.cc"],
    headers = ["matrix.h", "la.h", "uselapack.h", "clapack.h", "blas.h"],
    tests = ["uselapack_test.cc"],
    deplibs = ["fastlib/base:base", "fastlib/col:col",
               "fastlib/mmanager:mmapmm", ":netlib_blas", ":netlib_lapack"])
