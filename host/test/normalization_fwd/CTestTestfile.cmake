# CMake generated Testfile for 
# Source directory: /root/workspace/composable_kernel/test/normalization_fwd
# Build directory: /root/workspace/composable_kernel/host/test/normalization_fwd
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_layernorm2d_fwd_fp32 "/root/workspace/composable_kernel/host/bin/test_layernorm2d_fwd_fp32")
set_tests_properties(test_layernorm2d_fwd_fp32 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;2;add_gtest_executable;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;0;")
add_test(test_groupnorm_fwd_fp32 "/root/workspace/composable_kernel/host/bin/test_groupnorm_fwd_fp32")
set_tests_properties(test_groupnorm_fwd_fp32 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;8;add_gtest_executable;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;0;")
add_test(test_layernorm2d_fwd_fp16 "/root/workspace/composable_kernel/host/bin/test_layernorm2d_fwd_fp16")
set_tests_properties(test_layernorm2d_fwd_fp16 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;14;add_gtest_executable;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;0;")
add_test(test_layernorm4d_fwd_fp16 "/root/workspace/composable_kernel/host/bin/test_layernorm4d_fwd_fp16")
set_tests_properties(test_layernorm4d_fwd_fp16 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;20;add_gtest_executable;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;0;")
add_test(test_groupnorm_fwd_fp16 "/root/workspace/composable_kernel/host/bin/test_groupnorm_fwd_fp16")
set_tests_properties(test_groupnorm_fwd_fp16 PROPERTIES  _BACKTRACE_TRIPLES "/root/workspace/composable_kernel/test/CMakeLists.txt;113;add_test;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;26;add_gtest_executable;/root/workspace/composable_kernel/test/normalization_fwd/CMakeLists.txt;0;")
