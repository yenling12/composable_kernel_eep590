# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/grouped_convnd_fwd
# Build directory: /root/workspace/composable_kernel/host/test/grouped_convnd_fwd
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_grouped_convnd_fwd "/root/workspace/composable_kernel/host/bin/test_grouped_convnd_fwd")
set_tests_properties(test_grouped_convnd_fwd PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/grouped_convnd_fwd/CMakeLists.txt;1;add_gtest_executable;/root/workspace/composable_kernel/test/grouped_convnd_fwd/CMakeLists.txt;0;")
add_test(test_grouped_convnd_fwd_multi_ab_interface "/root/workspace/composable_kernel/host/bin/test_grouped_convnd_fwd_multi_ab_interface")
set_tests_properties(test_grouped_convnd_fwd_multi_ab_interface PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/grouped_convnd_fwd/CMakeLists.txt;4;add_gtest_executable;/root/workspace/composable_kernel/test/grouped_convnd_fwd/CMakeLists.txt;0;")
add_test(test_grouped_convnd_fwd_multi_d_interface_compatibility "/root/workspace/composable_kernel/host/bin/test_grouped_convnd_fwd_multi_d_interface_compatibility")
set_tests_properties(test_grouped_convnd_fwd_multi_d_interface_compatibility PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/grouped_convnd_fwd/CMakeLists.txt;7;add_gtest_executable;/root/workspace/composable_kernel/test/grouped_convnd_fwd/CMakeLists.txt;0;")
