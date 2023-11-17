# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/conv_tensor_rearrange
# Build directory: /root/workspace/composable_kernel/host/test/conv_tensor_rearrange
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_conv_tensor_rearrange "/root/workspace/composable_kernel/host/bin/test_conv_tensor_rearrange")
set_tests_properties(test_conv_tensor_rearrange PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/conv_tensor_rearrange/CMakeLists.txt;1;add_gtest_executable;/root/workspace/composable_kernel/test/conv_tensor_rearrange/CMakeLists.txt;0;")
add_test(test_conv_tensor_rearrange_interface "/root/workspace/composable_kernel/host/bin/test_conv_tensor_rearrange_interface")
set_tests_properties(test_conv_tensor_rearrange_interface PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/conv_tensor_rearrange/CMakeLists.txt;4;add_gtest_executable;/root/workspace/composable_kernel/test/conv_tensor_rearrange/CMakeLists.txt;0;")
