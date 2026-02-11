wget https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h

이후 app_server.cpp에서
'''text
#include "httplib.h"
'''

CMakeLists.txt에
'''text
include_directories(${CMAKE_SOURCE_DIR}/third_party)
'''
