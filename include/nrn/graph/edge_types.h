#pragma once

namespace nrn {

/// How an edge group is triggered during simulation.
enum class EdgeTrigger {
    Spike,       ///< Spike-triggered: I_syn += W @ spikes (chemical synapses).
    Continuous,  ///< Continuous: I += g * (V_pre - V_post) every step (gap junctions).
};

/// How the connectivity data is stored.
enum class EdgeRepresentation {
    Dense,        ///< Full [N_tgt, N_src] matrix — best for small, dense circuits.
    BlockSparse,  ///< CSR-of-dense-blocks — the default masked-dense paradigm.
    Auto,         ///< Let graph_compile choose based on density and size.
};

} // namespace nrn
