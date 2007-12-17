# Scroll down to see the main part.
# (This first part is dedicated to compiling and installing LAPACK.)

wgetrule(
    name = "trilinos_tgz",
    type = Types.ANY,
    url = "http://www.cc.gatech.edu/~gtg739c/trilinos-8.0.4.tar.gz")

def gen_compile_trilinos(sysentry, files, params):
  trilinos_tgz = files["trilinos_tgz"].single(Types.ANY)
  libtrilinospack = sysentry.file("KEEP/libtrilinospack.a", "arch", "kernel", "compiler")
  workspace_dir = os.path.join(os.path.dirname(libtrilinospack.name), "libtrilinospack_workspace")
  compiler_info = compilers[params["compiler"]]
  compiler = compiler_info.compiler_program("f")
  # Make sure we won't rm -rf anything bad
  assert "libtrilinospack_workspace" in workspace_dir
  sysentry.command("echo '... Extracting Source trilinos files...'")
  sysentry.command("mkdir -p %s" % sq(workspace_dir))
  sysentry.command("cd %s && tar -xzf %s" % (sq(workspace_dir), sq(trilinos_tgz.name)))
  sysentry.command("rm -rf %s"  %  sq(trilinos_tgz.name))
  sysentry.command("echo '*** Compiling Trilinos 8.0.4 version.'")
  sysentry.command("echo '... We only compile Epetra, Teuchos, Anasazi:'")
  sysentry.command("echo '... At this time we do not support distributed trilinos.'")
  sysentry.command("echo '... Only serial version is built.'")
  sysentry.command("echo '... For more information see http://trilinos.sandia.gov/Trilinos8.0Tutorial.pdf'")
  sysentry.command("echo '... This may take several minutes .'")
  sysentry.command("cd %s/trilinos-8.0.4 && mkdir -p LINUX_SERIAL " % sq(workspace_dir))
  sysentry.command("echo '***Configuring Trilinos...'")
  sysentry.command("cd %s/trilinos-8.0.4/LINUX_SERIAL && \
	                  ../configure --prefix=%s/trilinos-8.0.4/LINUX_SERIAL \
			              --disable-default-packages \
			              --enable-epetra \
										--enable-teuchos \
										--enable-anasazi \
										| tee configure_LINUX_SERIAL.log" % (sq(workspace_dir) ,
										                                     sq(workspace_dir)))
  sysentry.command("echo '...Configuration Done, if you encountered errors check\
			              logfile configure_LINUX_SERIAL.log'")
  sysentry.command("echo '*** Ready to compile now'");
  sysentry.command("cd %s/trilinos-8.0.4/LINUX_SERIAL && \
		              make everything| tee make_LINUX_SERIAL.log" % (sq(workspace_dir)))
  sysentry.command("cd %s/trilinos-8.0.4/LINUX_SERIAL && \
			              make install | tee make_install.LINUX_SERIAL.log" % (sq(workspace_dir)))
  sysentry.command("echo '*** Finished compiling, for errors check \
	                  logfile make_LINUX_SERIAL.log and make_install_LINUX_SERIAL.log'")
  #we have to find out how to combine all libraries into one but
	#first we have to remove any old libtrilinos.a
  sysentry.command("rm -f %s" % sq(libtrilinospack.name))
  sysentry.command("cd %s/trilinos-8.0.4/LINUX_SERIAL/lib && \
			              ar -x libepetra.a && \
			              ar -x libteuchos.a && \
			              ar -x libanasazi.a && \
										ar -rs %s *.o" % 
										(sq(workspace_dir), sq(libtrilinospack.name)))
  sysentry.command("echo '... Created archive'")
  sysentry.command("echo '... Copying the include header files to fastlib'")
  sysentry.command("cp -r %s/trilinos-8.0.4/LINUX_SERIAL/include ." % sq(workspace_dir))
  sysentry.command("echo '... Cleaning'")
  sysentry.command("rm -rf %s" % sq(workspace_dir))
  sysentry.command("echo '*** Done with TRILINOS!'")
  return [(Types.LINKABLE, libtrilinospack)]

customrule(
    name = "libtrilinos",
    dependencies = {"trilinos_tgz": [find(":trilinos_tgz")]},
    doit_fn = gen_compile_trilinos)

#---- This is the test part of the installation 

librule(name = "trilinos",
		    headers = lglob("trilinons/include/*.hpp"),
				deplibs =["la:libblaspack",":libtrilinos"])

binrule(
		name = "test_trilinos",
		#cflags="-I./include/",
		sources = ["test.cc"],
		deplibs = ["base:base", ":trilinos"])

