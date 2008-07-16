librule(name="disk_allnn",
    headers=["disk_allnn.h"],
    tests=["disk_allnn_test.cc"],
    deplibs=["fastlib:fastlib", "fastlib/mmanager:mmapmm"]
    );
binrule(name="main",
    sources=["main.cc"],
    deplibs=[":disk_allnn"]);
