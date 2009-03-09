# Scroll down to see the main part.
import commands;
import sys

wgetrule(
    name = "optpp_lock",
    type = Types.ANY,
    url = "www.gatech.edu/optpp.lock")

def make_optpp(sysentry, files, params):
  libblas = sysentry.file("KEEP/libblas.a", "arch", "kernel", "compiler")
  liboptpp = sysentry.file("KEEP/libopt.a", "arch", "kernel", "compiler")
  libnewmat = sysentry.file("KEEP/libnewmat.a", "arch", "kernel", "compiler")
  optpp_lock = files["optpp_lock"].single(Types.ANY)
  if sq(commands.getoutput("ls " + sq(optpp_lock.name)))==sq(optpp_lock.name):
    return [(Types.LINKABLE, liboptpp)]
  else:
    print "First time installing opt++, Here are your options:"
    print "1)Press 1 if you want opt++ to download and install it from sandia website"
    print "2)Press 2 if you think you have a better version already installed press 2"
    print "  You will be asked to give the full path along with the library name (/path/libexample.a)"
    print "  if after giving the path you realize that it is wrong or it doesn't work"
    print "  delete the $instalation_dir/fastlib/optimization/opt++/optpp.lock file and build your code again"
    resp=0;
    while resp!=1 and resp!=2:
      resp=input("Give me your choice now: ");
      if resp!=1 and resp!=2:
        print "You chose "+str(resp)+ " which is not a valid choice"
        commands.getoutput("rm -f " + sq(liboptpp.name));

      if resp==2:
        optpp_lib=input("Give me now the full path with the optpp library name in quotes ie \"/usr/lib/libopt.a\": ");
        print "I am creating a symbolic link to "+ str(optpp_lib)
        commands.getoutput("rm -f " + sq(liboptpp.name));
        print commands.getoutput("ln -s "+str(optpp_lib)+ " "+ sq(liboptpp.name));
        newmat_lib=input("Give me now the full path with the newmat library name in quotes ie \"/usr/lib/libnewmat.a\": ");
        print "I am creating a symbolic link to "+ str(newmat_lib)
        commands.getoutput("rm -f " + sq(libnewmat.name));
        print commands.getoutput("ln -s "+str(newmat_lib)+ " "+ sq(libnewmat.name));

      else:
        if resp==1:
          print "Now I will download opt++ from http://csmr.ca.sandia.gov/opt++/"
          workspace_dir = os.path.join(os.path.dirname(liboptpp.name), "optpp_workspace")
          print sq(workspace_dir)
          # Make sure we won't rm -rf anything bad
          assert "optpp_workspace" in workspace_dir
          sysentry.command("echo '... creating temporary workspace...'")
          sysentry.command("mkdir -p %s" % sq(workspace_dir))

          if os.path.basename(params["downloader"]) == "wget":
            sysentry.command("cd %s && %s http://csmr.ca.sandia.gov/opt++/optpp-2.4.tar.gz " % ((sq(workspace_dir)), sq(params["downloader"])))
            print "cd %s && %s http://csmr.ca.sandia.gov/opt++/optpp-2.4.tar.gz " % ((sq(workspace_dir)), sq(params["downloader"]))
            
          else:
            sysentry.command("cd %s && %s http://csmr.ca.sandia.gov/opt++/optpp-2.4.tar.gz -o optpp-2.4.tar.gz" % ((sq(workspace_dir)), sq(params["downloader"])))
          sysentry.command("cd %s && tar -xzf %s" % (sq(workspace_dir), sq("optpp-2.4.tar.gz")))
          compiler_info = compilers[params["compiler"]]
          compiler = compiler_info.compiler_program("f")
          sysentry.command("echo '*** Compiling  opt++ from sandia.gov'")
          sysentry.command("echo '... This may take several minutes '")
          sysentry.command("cd %s/optpp-2.4 && mkdir install_dir" % (sq(workspace_dir)))
          sysentry.command("cd %s/optpp-2.4 && ./configure --prefix=%s/install_dir --with-blas=%s" % (sq(workspace_dir),
            sq(workspace_dir), sq(libblas.name)))
          sysentry.command("cd %s/optpp-2.4 && make install" % (sq(workspace_dir)))
          sysentry.command("echo '... Almost done with opt++, now coping the libraries to bin_keep...'")
          sysentry.command("cd %s/install_dir/lib && mv libopt.a %s && mv libnewmat.a %s" % 
            (sq(workspace_dir), sq(liboptpp.name), sq(libnewmat.name)))
          sysentry.command("mv %s/install_dir/include ./include  " % 
            (sq(workspace_dir)))
          print "mv %s/install_dir/include ./include  " % (sq(workspace_dir))
          sysentry.command("echo '...  cleaning up.'")
          sysentry.command("rm -rf %s" % sq(workspace_dir))
          # Export it if the user requested.
      if params["prefix"] != "":
        destination_parent_dir = params["prefix"].rstrip("/").rstrip(os.sep) + "/lib/"
        sysentry.command("mkdir -p " + destination_parent_dir)
        sysentry.command("cp -f " + (sq(liboptpp.name)) + " " + destination_parent_dir)
        sysentry.command("cp -f " + (sq(libnewmat.name)) + " " + destination_parent_dir)
        sysentry.command("echo '*** Done with  opt++!'")
  print "Generating a optpp.lock file, if you want to reinstall differently delete it"
  print commands.getoutput("echo lock"  + " >"+sq(optpp_lock.name));
  return [(Types.LINKABLE, liboptpp)]

def make_newmat(sysentry, files, params):
  libnewmat = sysentry.file("KEEP/libnewmat.a", "arch", "kernel", "compiler")
  return [(Types.LINKABLE, libnewmat)]

customrule(
    name="optpp_install",
    dependencies = {"optpp_lock": [find(":optpp_lock")] },
    doit_fn = make_optpp
    )

# This is just a helper custom rule to return the library for the 
# newmat library that contains the matrix structures for opt++
customrule(
    name="newmat_install",
    dependencies = {"optpp_lock": [find(":optpp_lock")] },
    doit_fn = make_newmat
    )


librule(name="opt++",
  headers=["optimizer.h"],
  tests=["optimizer_test.cc","optimizer_tests.cc"],
  deplibs=["fastlib:fastlib", ":optpp_install", ":newmat_install"]
)
