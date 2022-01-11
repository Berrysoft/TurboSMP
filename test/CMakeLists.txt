function(fsmp_add_test name)
  add_executable(${name} "${name}.cu")
  target_link_libraries(${name} PRIVATE CUDA::cudart Boost::disable_autolinking Boost::dynamic_linking Boost::unit_test_framework)
  target_include_directories(${name} PUBLIC ../include)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

fsmp_add_test(dot)
