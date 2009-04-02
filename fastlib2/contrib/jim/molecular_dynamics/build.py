
librule(
    name = "particle",                  
    headers = ["particle_tree.h", "two_body_stat.h",
               "multi_physics_system.h", "force_error.h"],   
    deplibs = ["fastlib:fastlib", "fastlib:fastlib_int"]
    )

librule(
    name = "particle2",                  
    headers = ["particle_tree.h",
               "dual_physics_system.h"],   
    deplibs = ["fastlib:fastlib", "fastlib:fastlib_int"]
    )


librule(
    name = "test1",
    headers = ["particle_tree.h", "test_particle_stat.h", "two_body_stat.h"],
    deplibs = ["fastlib:fastlib", "fastlib:fastlib_int"]
    )


binrule(
    name = "simulation",
    sources = ["simulation_driver.cc"],
    deplibs = [":particle"]
    )

binrule(
    name = "test_stat",
    sources = ["test_particle_stat.cc"],
    deplibs = [":test1"]
    )

binrule(
    name = "dual_simulation",
    sources = ["dual_simulation_driver.cc"],
    deplibs = [":particle2"]
    )
