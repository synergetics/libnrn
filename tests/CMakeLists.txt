function(nrn_add_test test_name test_source)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE nrn GTest::gtest GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

nrn_add_test(test_types                  core/test_types.cpp)
nrn_add_test(test_state                  core/test_state.cpp)
nrn_add_test(test_module                 core/test_module.cpp)

nrn_add_test(test_lif                    neuron/test_lif.cpp)
nrn_add_test(test_adex                   neuron/test_adex.cpp)
nrn_add_test(test_izhikevich             neuron/test_izhikevich.cpp)

nrn_add_test(test_connectivity_tensor    connectivity/test_connectivity_tensor.cpp)
nrn_add_test(test_topology               connectivity/test_topology.cpp)

nrn_add_test(test_stdp                   plasticity/test_stdp.cpp)

nrn_add_test(test_clock                  simulation/test_clock.cpp)
nrn_add_test(test_spike_buffer           simulation/test_spike_buffer.cpp)

nrn_add_test(test_brunel_network         integration/test_brunel_network.cpp)
