librule(name="obj_test",
        headers=["obj_grad_hess_class.h"],
        sources=["obj_grad_hess_class.cc"],
        deplibs=["fastlib::fastlib"])

binrule(name="obj_grad_hess_main",
        sources=["obj_grad_hess_main.cc"],
        deplibs=[":obj_test"])

