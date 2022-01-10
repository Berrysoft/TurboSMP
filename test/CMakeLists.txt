set(Boost_USE_STATIC_LIBS ON)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

function(fsmp_add_test name)
  add_executable(${name} "${name}.cu")
  target_link_libraries(${name} PRIVATE CUDA::cudart Boost::unit_test_framework)
  target_include_directories(${name} PUBLIC ../include)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

fsmp_add_test(dot)
