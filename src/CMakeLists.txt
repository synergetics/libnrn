set(NRN_SOURCES
    core/types.cpp
    core/state.cpp
    core/module.cpp

    neuron/lif.cpp
    neuron/adex.cpp
    neuron/izhikevich.cpp

    synapse/static.cpp
    synapse/exponential.cpp
    synapse/double_exponential.cpp
    synapse/ampa.cpp

    connectivity/connectivity_tensor.cpp
    connectivity/connect.cpp
    connectivity/topology/random.cpp
    connectivity/topology/distance_dependent.cpp

    plasticity/plasticity_rule.cpp
    plasticity/stdp.cpp

    network/population.cpp
    network/region.cpp

    simulation/clock.cpp
    simulation/spike_buffer.cpp
    simulation/runner.cpp

    monitor/recorder.cpp
    monitor/spike_recorder.cpp
    monitor/state_recorder.cpp

    config/config.cpp
)

set(NRN_CUDA_SOURCES
    neuron/kernels/lif_kernel.cu
    neuron/kernels/adex_kernel.cu
    neuron/kernels/izhikevich_kernel.cu
    synapse/kernels/synapse_kernels.cu
)

add_library(nrn SHARED ${NRN_SOURCES} ${NRN_CUDA_SOURCES})

target_include_directories(nrn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(nrn
    PUBLIC
        ${TORCH_LIBRARIES}
    PRIVATE
        yaml-cpp::yaml-cpp
)

set_target_properties(nrn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${NRN_CUDA_ARCHITECTURES}"
)

install(TARGETS nrn DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/nrn DESTINATION include)
