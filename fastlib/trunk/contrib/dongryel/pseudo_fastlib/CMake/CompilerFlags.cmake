
set(CMAKE_C_FLAGS "-Wall")

set(CMAKE_CXX_FLAGS "-Wall -Woverloaded-virtual -fno-exceptions -Wparentheses")
set(CMAKE_CXX_FLAGS_DEBUG "g3 -DDEBUG -O0")
set(CMAKE_CXX_FLAGS_RELEASE 
    "-O2 -finline-functions -finline-limit=2000 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_VERBOSE "-g -DDEBUG -DVERBOSE")
set(CMAKE_CXX_FLAGS_FAST "O2 -finline-functions -finline-limit=2000 -fomit-frame-pointer -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_UNSAFE "-O3 -ffast-math -g -fomit-frame-pointer -DNDEBUG")
set(CMAKE_CXX_FLAGS_PROFILE "-O2 -pg -finline-limit=12 -DPROFILE -DNDEBUG")
set(CMAKE_CXX_FLAGS_SMALL "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_TRACE "-g -DNDEBUG")

add_definitions( -DDISABLE_DISK_MATRIX )

