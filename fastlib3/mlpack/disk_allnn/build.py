librule(name="disk_allnn",
    headers=["disk_allnn.h"],
    tests=["disk_allnn_test.cc"],
    deplibs=["fastlib:fastlib", "fastlib/mmanager:mmapmm",
             "fastlib/tree:tree"]
    );
binrule(name="main",
    sources=["main.cc"],
    deplibs=[":disk_allnn"]);

binrule(name="mainp",
    sources=["main.cc"],
    cflags="-DPREFETCH",
    deplibs=[":disk_allnn"]);
