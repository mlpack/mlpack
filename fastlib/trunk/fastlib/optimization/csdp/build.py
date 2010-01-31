# Scroll down to see the main part.
import commands;
import sys

wgetrule(
    name = "csdp_lock",
    type = Types.ANY,
    url = "www.gatech.edu/csdp.lock")

def make_csdp(sysentry, files, params):
  libcsdp = sysentry.file("KEEP/libsdp.a", "arch", "kernel", "compiler")
  optpp_lock = files["optpp_lock"].single(Types.ANY)
  if sq(commands.getoutput("ls " + sq(csdp_lock.name)))==sq(csdp_lock.name):
    return [(Types.LINKABLE, libcsdp)]
  else:
    print "First time installing CSDP, Here are your options:"
    print "1)Press 1 if you want CSDP to download and install it from COIN-OR website"
    print "2)Press 2 if you think you have a better version already installed press 2"
    print "  You will be asked to give the full path along with the library name (/path/libexample.a)"
    print "  if after giving the path you realize that it is wrong or it doesn't work"
    print "  delete the $instalation_dir/fastlib/optimization/csdp/csdp.lock file and build your code again"
    resp=0;
    while resp!=1 and resp!=2:
      resp=input("Give me your choice now: ");
      if resp!=1 and resp!=2:
        print "You chose "+str(resp)+ " which is not a valid choice"
        commands.getoutput("rm -f " + sq(libcsdp.name));

      if resp==2:
        csdp_lib=input("Give me now the full path with the csdp library name in quotes ie \"/usr/lib/libsdp.a\": ");
        print "I am creating a symbolic link to "+ str(csdp_lib)
        commands.getoutput("rm -f " + sq(libcsdp.name));
        print commands.getoutput("ln -s "+str(csdp_lib)+ " "+ sq(libcsdp.name));
      else:
        if resp==1:
          print "Now I will download opt++ from http://www.coin-or.org/Tarballs/Csdp/Csdp-6.0.1.tgz"
          workspace_dir = os.path.join(os.path.dirname(liboptpp.name), "csdp_workspace")
          print sq(workspace_dir)
          # Make sure we won't rm -rf anything bad
          assert "csdp_workspace" in workspace_dir
          sysentry.command("echo '... creating temporary workspace...'")
          sysentry.command("mkdir -p %s" % sq(workspace_dir))

          if os.path.basename(params["downloader"]) == "wget":
            sysentry.command("cd %s && %s http://www.coin-or.org/Tarballs/Csdp/Csdp-6.0.1.tgz " % ((sq(workspace_dir)), sq(params["downloader"])))
          else:
            sysentry.command("cd %s && %s http://www.coin-or.org/Tarballs/Csdp/Csdp-6.0.1.tgz -o Csdp-6.0.1.tgz" % ((sq(workspace_dir)), sq(params["downloader"])))
          sysentry.command("cd %s && tar -xzf %s" % (sq(workspace_dir), sq("Csdp-6.0.1.tgz")))
          compiler_info = compilers[params["compiler"]]
          compiler = compiler_info.compiler_program("f")
          sysentry.command("echo '*** Compiling  CSDP from COIN-OR'")
          sysentry.command("echo '... This may take several minutes '")
          sysentry.command("cd %s/Csdp-6.0.1/lib && make" % (sq(workspace_dir)))
          sysentry.command("echo '... Almost done with CSDP, now coping the libraries to bin_keep...'")
          sysentry.command("cd %s/Csdp-6.0.1/lib && mv libsdp.a %s" % 
            (sq(workspace_dir), sq(libcsdp.name) ))
          sysentry.command("mv %s/Csdp-6.0.1/include ./include  " % 
            (sq(workspace_dir)))
          sysentry.command("echo '...  cleaning up.'")
          sysentry.command("rm -rf %s" % sq(workspace_dir))
          # Export it if the user requested.
      if params["prefix"] != "":
        destination_parent_dir = params["prefix"].rstrip("/").rstrip(os.sep) + "/lib/"
        sysentry.command("mkdir -p " + destination_parent_dir)
        sysentry.command("cp -f " + (sq(libcsdp.name)) + " " + destination_parent_dir)
        sysentry.command("echo '*** Done with  CSDP!'")
  print "Generating a csdp.lock file, if you want to reinstall differently delete it"
  print commands.getoutput("echo lock"  + " >"+sq(csdp_lock.name));
  return [(Types.LINKABLE, libcsdp)]


customrule(
    name="csdp_install",
    dependencies = {"csdp_lock": [find(":csdp_lock")] },
    doit_fn = make_newmat
    )


librule(name="csdp",
  headers=["optimizer.h"],
  tests=["optimizer_test.cc"],
  deplibs=["fastlib:fastlib", ":csdp_install"]
)
