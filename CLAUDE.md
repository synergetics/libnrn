# libnrn — Biological Neural Simulation Framework

## Vision

A GPU-native, libtorch-semantic framework for simulating the full biological neural
substrate — neurons, glia, neuromodulation, structural plasticity — at scale on
NVIDIA GPU clusters.

**No existing system does this.** NEST GPU scales spiking networks to 3.37 billion
neurons on 12,000 GPUs but ignores glia. NEST CPU supports astrocytes but is CPU-only.
Brian2CUDA can run glia on a single GPU. None use modern interconnects (NVSHMEM, NCCL).
The 2024-2025 discovery of astroengrams (astrocyte memory traces) makes neuron-only
simulation demonstrably incomplete.

libnrn fills this gap with a from-scratch design that rethinks computational primitives
for GPU efficiency rather than literally transcribing biological network topology onto
hardware that hates irregular sparse access patterns.

---

## Part I: Computational Philosophy

### The Core Problem

Traditional neural simulators model biological networks literally: each neuron is an
object, each synapse is a pointer, connectivity is an adjacency list. This is an
anti-pattern for GPUs — irregular memory access, pointer chasing, thread divergence,
low occupancy.

Deep learning frameworks solved a similar problem by representing everything as dense
tensor operations. libnrn applies this insight to biological simulation.

### The Masked-Dense Paradigm

Instead of literal sparse graph connectivity, represent networks as:

```
effective_connectivity = W ⊙ M_structural ⊙ M_modulatory ⊙ M_stochastic
output = f(effective_connectivity @ input_state)
```

Where:
- `W` is a dense (or block-dense) weight/efficacy tensor
- `M_structural` is a binary/sparse mask encoding which connections exist (static or
  slowly changing — synaptogenesis/elimination timescale)
- `M_modulatory` is a continuous mask encoding neuromodulatory gain (dopamine, serotonin
  etc. — seconds to minutes timescale)
- `M_stochastic` is a per-step random mask for synaptic release probability, channel
  noise, etc.
- `⊙` is element-wise (Hadamard) product

**This converts irregular sparse computation into regular dense computation with
masking** — exactly what GPUs and tensor cores are optimized for.

### Block-Sparse Hierarchy

Full density is impossible at scale (10^6 neurons = 10^12 matrix entries = 4 TB). The
solution is **hierarchical block-sparse with dense blocks**:

```
Level 0: Full network     — sparse block structure (which populations connect)
Level 1: Population pairs — dense blocks (e.g., 256×256 or 1024×1024)
Level 2: Within blocks    — dense matmul with mask overlays
```

This maps naturally to:
- GPU thread blocks process one dense block each
- Tensor cores handle the dense matmul within blocks
- Block-sparse structure between populations uses CSR-of-dense-blocks
- cuSPARSE blocked formats or custom block-sparse kernels

Biologically, this hierarchy corresponds to:
- Level 0: Brain regions / cortical areas (sparse inter-region connectivity)
- Level 1: Neural populations / layers / minicolumns (denser intra-region)
- Level 2: Local circuits (dense local connectivity with distance-dependent masking)

### Handling Temporality and Recurrence

Three key insights from modern ML for handling time on GPUs:

**1. Parallel Scan for State Evolution**

Instead of sequential time stepping `x(t+1) = A·x(t) + B·u(t)`, use parallel prefix
scan (associative scan) to compute all time steps simultaneously. This is how S4/Mamba
handle recurrence:

For linear dynamics: $x(t) = A^t · x(0) + Σ A^(t-k) · B · u(k)$
Parallel scan computes this in O(log T) parallel steps instead of O(T) sequential

For nonlinear dynamics (most neuron models), use operator splitting:
- Linear part (leak, passive membrane): parallel scan
- Nonlinear part (spike threshold, ion channels): pointwise parallel

**2. Batched Time Windows**

Rather than stepping one dt at a time, process windows of W timesteps as a batch:
- State tensor shape: `[batch_time=W, neurons, state_vars]`
- Spike delays become shifts within the time dimension
- Recurrence within the window uses scan; between windows is sequential
- Window size W tuned to balance parallelism vs. memory

**3. Event Compression**

Spikes are sparse in time (~0.01-0.1% of neurons spike per 0.1ms step). Rather than
processing all neurons every step:
- Maintain a compact spike event buffer (neuron_id, time) — not a dense binary matrix
- Use scatter operations for post-synaptic current delivery
- Separate "active" and "quiescent" neuron processing paths

### What This Means Architecturally

The framework operates on tensors, not graphs. A "network" is:
- A state tensor `S ∈ R^{N × D}` (N neurons/cells, D state dimensions)
- A set of connectivity tensors (block-sparse with dense blocks + mask overlays)
- A set of temporal operators (parallel scan coefficients, delay buffers)
- A set of plasticity rules operating on the connectivity tensors

This is why the libtorch analogy is deep — it's not just API similarity, it's the
same computational substrate (tensor ops on GPU) applied to a different domain.

### When This Breaks Down

Honest assessment of where the dense/block-sparse paradigm has limits:

1. **Very long-range sparse connections** (e.g., a single neuron projecting across the
   brain): doesn't fit neatly into block structure. Solution: separate "long-range
   projection" sparse layer on top of block-dense local connectivity.

2. **Highly heterogeneous connectivity** (each neuron connects to a unique random
   subset): the mask becomes nearly random, losing block structure benefits. Solution:
   for truly random connectivity, use hashing-based approaches (like locality-sensitive
   hashing to approximate neighborhood structure).

3. **Dynamic structural plasticity**: changing which connections exist means updating
   masks, which is cheap for flipping bits but expensive if it changes block structure.
   Solution: over-provision blocks and use masks to enable/disable; only restructure
   blocks on slow timescales.

4. **Multi-compartment neurons**: dendritic trees are genuinely tree-structured, not
   matrix-structured. Solution: within-neuron computation uses tree-parallel algorithms
   (Hines); between-neuron uses block-dense. Two separate compute phases.

5. **Scale > 10^9 neurons**: even block-sparse becomes enormous. Solution: procedural
   connectivity (GeNN-style) — compute connectivity on-the-fly from a function rather
   than storing it.

---

## Part II: Abstractions & API

### Mapping to libtorch

| libtorch                | libnrn             | Purpose                                            |
| ----------------------- | ------------------ | -------------------------------------------------- |
| `torch::Tensor`         | `nrn::Tensor`      | State tensors (voltages, currents, concentrations) |
| `torch::nn::Module`     | `nrn::Module`      | Composable simulation components                   |
| `torch::autograd`       | `nrn::Plasticity`  | Local/global learning rules (replaces backprop)    |
| `torch::optim`          | `nrn::Optimizer`   | Meta-learning / rule parameter updates             |
| `torch::nn::Sequential` | `nrn::Network`     | Composed networks with connectivity                |
| `torch::distributed`    | `nrn::Distributed` | Multi-GPU/node via NCCL + NVSHMEM                  |
| `torch::data`           | `nrn::Stimulus`    | Input generators and replay buffers                |
| `torch::jit`/`compile`  | `nrn::Compile`     | Kernel fusion / graph optimization                 |
| `torch::Device`         | `nrn::Device`      | GPU placement, partitioning                        |
| TensorBoard             | `nrn::Monitor`     | Observation, recording, analysis                   |

### Module System

```cpp
class Module {
  // Core simulation interface
  virtual void forward(nrn::State& state, nrn::Time t, nrn::Duration dt) = 0;
  virtual void reset() = 0;

  // Torch-style parameter management
  void register_parameter(std::string name, nrn::Tensor param);
  void register_buffer(std::string name, nrn::Tensor buf);
  void register_mask(std::string name, nrn::Mask mask);  // connectivity masks

  // Serialization
  StateDict state_dict();
  void load_state_dict(StateDict dict);

  // Composition
  void register_module(std::string name, std::shared_ptr<Module> module);

  // Device management
  void to(nrn::Device device);
  void to(nrn::dtype dtype);
};
```

Modules compose: `Population` is a `Module`, `Network` (populations + connectivity) is
a `Module`, `Region` (networks + glia + modulators) is a `Module`, `Brain` (regions)
is a `Module`.

### C++ API Example

```cpp
// Create neuron populations
auto exc = nrn::neuron::AdEx(80000, nrn::AdExOptions()
    .v_rest(-70.6_mV).v_thresh(-50.4_mV).tau_m(20.0_ms)
    .a(4.0_nS).b(0.0805_nA).delta_t(2.0_mV));

auto inh = nrn::neuron::LIF(20000, nrn::LIFOptions()
    .v_rest(-65.0_mV).v_thresh(-50.0_mV).tau_m(10.0_ms));

// Create connectivity with block-dense + mask paradigm
auto ee_conn = nrn::connect(exc, exc,
    nrn::topology::DistanceDependent(/*sigma=*/200.0_um),
    nrn::synapse::AMPA().tau_rise(0.5_ms).tau_decay(5.0_ms),
    nrn::ConnectOptions().block_size(512).representation(nrn::BlockDense));

// Attach plasticity
ee_conn->attach(nrn::plasticity::STDP()
    .tau_plus(20.0_ms).tau_minus(20.0_ms)
    .a_plus(0.01).a_minus(-0.012));

// Create astrocyte population
auto astro = nrn::glia::Astrocyte(15000, nrn::AstrocyteOptions()
    .model(nrn::astro::GChI)
    .territory_radius(50.0_um));

// Tripartite connectivity
auto tripartite = nrn::tripartite_connect(ee_conn, astro,
    nrn::TripartiteOptions()
        .territory(nrn::territory::Voronoi)
        .neuron_to_astro(nrn::coupling::GlutamateSpillover(0.1))
        .astro_to_synapse(nrn::coupling::Gliotransmission())
        .astro_to_neuron(nrn::coupling::SIC(/*threshold=*/0.5_uM)));

// Neuromodulatory system (global overlay mask)
auto dopamine = nrn::modulator::Dopaminergic(nrn::DopamineOptions()
    .baseline(1.0_nM).release_rate(10.0_nM_per_spike)
    .reuptake_tau(200.0_ms));

// Compose into a region
auto region = nrn::Region("cortical_column");
region->add(exc, inh, astro);
region->add(ee_conn, ei_conn, ie_conn, ii_conn);
region->add(tripartite);
region->add(dopamine);

// Simulation
auto sim = nrn::Simulation(region, nrn::SimOptions()
    .dt_fast(0.1_ms)        // neuron/synapse timestep
    .dt_slow(10.0_ms)       // astrocyte/modulator timestep
    .duration(10.0_s)
    .device(nrn::Device::GPU(0)));

// Recording
sim.record(exc, {"v", "spike"}, /*subsample=*/1000);
sim.record(astro, {"Ca", "IP3"}, /*subsample=*/100);

sim.run();
auto spikes = sim.get_spikes(exc);
sim.save("output.h5");
```

---

## Part III: Component Specifications

### 1. Neuron Models

Each model is a CUDA kernel (or kernel family). Two implementation strategies:

**Hand-optimized kernels** for standard models:

| Model                       | Type          | State Vars  | Complexity | Notes                         |
| --------------------------- | ------------- | ----------- | ---------- | ----------------------------- |
| Leaky Integrate-and-Fire    | Point         | 1 (v)       | Minimal    | Largest scale; parallel scan  |
| Adaptive Exponential (AdEx) | Point         | 2 (v, w)    | Low        | Default cortical neuron       |
| Izhikevich                  | Point         | 2 (v, u)    | Low        | Diverse firing patterns       |
| Exponential IF              | Point         | 1 (v)       | Minimal    | Efficient cortical proxy      |
| FitzHugh-Nagumo             | Point         | 2 (v, w)    | Low        | Oscillations, excitable media |
| Morris-Lecar                | Point         | 2 (v, w)    | Medium     | Bifurcation analysis          |
| Hodgkin-Huxley              | Compartmental | 4 (v,m,h,n) | High       | Biophysical detail            |
| Multi-compartment (generic) | Compartmental | N×D         | High       | Dendritic computation         |
| Poisson generator           | Stochastic    | 0           | Minimal    | Input layer / noise           |

**Configurable/parametric kernel** for custom models:
- User provides ODE equations (DSL or direct CUDA)
- Runtime compilation via NVRTC
- Parameter tensors control per-neuron behavior

**Configurable properties per neuron** (as parameter tensors, not per-object):
- Membrane: resting potential, threshold, reset, capacitance, leak conductance
- Ion channels: Na/K/Ca/Cl densities and kinetics (for HH-class models)
- Action potential: refractory period, afterhyperpolarization, spike shape
- Calcium dynamics: intracellular Ca, buffering, extrusion
- Morphology (multi-compartment): tree structure, compartment properties
- Stochasticity: channel noise amplitude, spontaneous rate

**Compartmental model support:**
- SWC/morphology import for realistic geometries
- Hines algorithm adapted for GPU (tree-parallel within neuron, data-parallel across)
- Separate compute phase from inter-neuron communication
- Dendritic spines as sub-compartments

### 2. Synapse Models

Synapses are **tensor elements** in the connectivity matrices, not objects. Each synapse
has state stored in parallel arrays (SoA):

| Type                  | State Variables    | Dynamics                                |
| --------------------- | ------------------ | --------------------------------------- |
| Static                | w                  | Fixed weight                            |
| Exponential decay     | w, I               | PSC with exponential kernel             |
| Double exponential    | w, I_rise, I_decay | Rise + decay time constants             |
| AMPA                  | w, g, s            | Fast excitatory, receptor kinetics      |
| NMDA                  | w, g, s, Mg_block  | Voltage-dependent, slow                 |
| GABA_A                | w, g, s            | Fast inhibitory                         |
| GABA_B                | w, g, s, G         | Slow inhibitory, G-protein              |
| Gap junction          | g                  | Bidirectional electrical                |
| Short-term plasticity | w, u, x, I         | Tsodyks-Markram facilitation/depression |
| Neuromodulated        | w, gain            | Gain modulated by modulator field       |

**Synapse representation in the block-dense paradigm:**
- Within a dense block: weight tensor `W[block_i, block_j]` of shape `[B, B]`
- Mask overlay: `M[block_i, block_j]` of shape `[B, B]` (binary or float)
- Additional state tensors (traces, STP state) parallel to W
- Delays: separate integer tensor `D[block_i, block_j]` indexing into ring buffer

### 3. Glial Cells

#### 3.1 Astrocytes (Phase 1 priority)

Astrocytes are the most computationally significant glial type and have first-class
support:

**Models:**

| Model       | State Variables   | ODEs | Use Case                    |
| ----------- | ----------------- | ---- | --------------------------- |
| Minimal SIC | Ca                | 1    | Maximum scale               |
| Li-Rinzel   | Ca, h             | 2    | Simplest calcium oscillator |
| G-ChI       | Ca, h, IP3        | 3    | Full tripartite synapse     |
| UAR         | Ca, h, IP3 + mito | 5+   | Metabolic coupling          |

**Key dynamics:**
- **Calcium signaling**: IP3-mediated Ca release, store-operated entry, calcium waves
  through gap junction networks
- **Tripartite synapse**: detect neurotransmitter spillover from ensheathed synapses,
  release gliotransmitters (glutamate, D-serine, ATP) that modulate synaptic transmission
- **Potassium buffering**: spatial K+ redistribution preventing extracellular accumulation
- **Territorial domains**: each astrocyte covers ~10^4-10^5 synapses (non-overlapping
  Voronoi-like territories)

**GPU implementation:**
- Astrocyte state tensor: `[N_astro, D_state]`
- Territory mapping: `territory_map[synapse_id] → astrocyte_id` (or block-level mapping)
- Operates on slow timescale (dt_slow, 1-100ms) via operator splitting
- Gap junction coupling as sparse matmul between astrocyte state vectors

**Astrocyte-neuron coupling:**
- Neuron→Astrocyte: accumulate spike counts over fast window, deliver as glutamate bolus
  at slow step boundary. Implemented as scatter-reduce over territory map.
- Astrocyte→Neuron: SIC emission when Ca crosses threshold. Scheduled into spike event
  queue for delivery during next fast window.
- Astrocyte→Synapse: gliotransmitter modulates the modulatory mask overlay `M_modulatory`
  for covered synapses.

#### 3.2 Oligodendrocytes (Phase 2)

- **Myelination dynamics**: myelin sheath formation/maintenance
- **Adaptive myelination**: activity-dependent changes to conduction velocity
- **Implementation**: modifies the delay tensor `D` for myelinated axonal segments
- **Saltatory conduction**: nodes of Ranvier as special compartments

#### 3.3 Microglia (Phase 2)

- **Synaptic pruning**: activity-dependent synapse elimination (modifies structural mask)
- **Complement tagging**: marks synapses for elimination based on activity statistics
- **Inflammatory state**: cytokine release modifying neuronal excitability (global modifier)
- **Surveillance**: territorial coverage, process motility (slow timescale)

#### 3.4 Other Glia (Phase 3+)

- NG2 / OPCs: synaptic input reception, proliferation/differentiation
- Ependymal cells: CSF flow (if modeling ventricular systems)

### 4. Connectivity

#### 4.1 The Block-Dense + Mask Architecture

Connectivity between populations A and B is represented as:

```
ConnectivityTensor {
  // Block structure (which blocks exist — sparse)
  BlockIndex block_map;         // CSR-like: which (A_block, B_block) pairs exist

  // Per-block dense data
  Tensor weights;               // [n_blocks, B, B] — synaptic efficacy
  Tensor structural_mask;       // [n_blocks, B, B] — which connections exist (binary)
  Tensor modulatory_mask;       // [n_blocks, B, B] — neuromodulatory gain (float)
  Tensor delays;                // [n_blocks, B, B] — axonal delay (int, indexes ring buf)

  // Per-block synapse state (for dynamic synapses)
  Tensor stp_u, stp_x;         // short-term plasticity state
  Tensor trace_pre, trace_post; // eligibility traces for plasticity
}
```

**Block size** is a tunable parameter (128, 256, 512, 1024). Smaller blocks = finer
granularity but less tensor core utilization. Larger blocks = more wasted computation
on masked-out entries but better hardware utilization. Sweet spot depends on connectivity
density and GPU architecture.

#### 4.2 Topology Generators

Built-in generators that produce block structure + initial masks:

| Pattern              | Block Structure                  | Mask Pattern                   |
| -------------------- | -------------------------------- | ------------------------------ |
| Random (Erdos-Renyi) | All blocks exist                 | Random binary, p per entry     |
| Distance-dependent   | Nearby blocks only               | Gaussian/exponential decay     |
| Small-world          | Local blocks + random long-range | Dense local, sparse random     |
| Scale-free           | Hub blocks fully connected       | Preferential attachment within |
| Columnar             | Intra-column blocks dense        | Layer-specific connectivity    |
| Laminar              | Inter-layer blocks               | Layer-specific rules           |
| Topographic          | Diagonal blocks                  | Ordered projection             |
| From connectome data | Data-driven                      | Data-driven                    |
| Custom               | User-defined                     | User-defined                   |

Additionally: **Procedural connectivity** for extreme scale — don't store the
connectivity tensor, compute `f(source_id, target_id) → {weight, delay, exists}`
on-the-fly within the kernel. Uses more compute but near-zero memory. Good for regular
patterns (nearest-neighbor, grid, all-to-all within column).

#### 4.3 Structural Plasticity

Structural changes operate on the **structural mask**:

- **Synaptogenesis**: flip 0→1 in structural mask (new connection). Criteria: activity
  correlation, spatial proximity, molecular cues (configurable rule).
- **Elimination**: flip 1→0 (remove connection). Criteria: low weight, low activity,
  microglial complement tagging.
- **Rewiring**: combination of elimination + synaptogenesis.
- **Block-level changes**: if an entire block becomes active/inactive, update the block
  map (rare, expensive operation — batched on slow timescale).

Operates on the **slowest timescale** (minutes to hours of simulated time). Mask updates
are cheap (bit flips on GPU). Block structure changes are batched.

#### 4.4 Connectivity Types

| Connection Type       | Implementation                                             |
| --------------------- | ---------------------------------------------------------- |
| Neuron→Neuron         | Block-dense + mask (primary paradigm)                      |
| Neuron→Astrocyte      | Territory map (scatter-reduce of spikes by territory)      |
| Astrocyte→Neuron      | Territory map inverse (SIC to covered neurons)             |
| Astrocyte→Synapse     | Territory map (modulate M_modulatory for covered syns)     |
| Astrocyte→Astrocyte   | Gap junction sparse matrix (small N, can be dense)         |
| Microglia→Synapse     | Complement tag map (modifies M_structural)                 |
| Oligodendrocyte→Axon  | Myelination map (modifies delay tensor D)                  |
| Neuromodulatory       | Broadcast/diffusion field (modifies M_modulatory globally) |
| Long-range projection | Explicit sparse (CSR) for cross-region connections         |

### 5. Learning & Plasticity

Plasticity rules are **pluggable modules** that operate on connectivity tensors:

```cpp
class PlasticityRule : public Module {
  virtual void update(
    ConnectivityTensor& conn,
    const Tensor& pre_state,
    const Tensor& post_state,
    const Tensor& modulator_state,  // neuromodulatory context
    Time t, Duration dt
  ) = 0;
};
```

#### 5.1 Local Learning Rules

| Rule                           | Operates On           | Timescale               |
| ------------------------------ | --------------------- | ----------------------- |
| STDP (pair-based)              | weights               | 10-100 ms               |
| STDP (triplet)                 | weights               | 10-100 ms               |
| Reward-modulated STDP (R-STDP) | weights × eligibility | ms (trace) + s (reward) |
| BCM                            | weights + threshold   | seconds                 |
| Oja's rule                     | weights (normalized)  | seconds                 |
| Voltage-dependent plasticity   | weights               | 10-100 ms               |
| Calcium-based plasticity       | weights               | 10-100 ms               |
| Homeostatic scaling            | all weights (global)  | minutes-hours           |
| Intrinsic plasticity           | neuron parameters     | minutes-hours           |
| Metaplasticity                 | plasticity thresholds | hours                   |
| Heterosynaptic plasticity      | neighbor weights      | seconds                 |

All rules are implemented as **tensor operations** on the connectivity tensor — no
per-synapse branching. Example STDP:

```
dw = A_plus * pre_trace * post_spike - A_minus * post_trace * pre_spike
w += learning_rate * dw * M_structural * M_modulatory  // only update existing, modulated synapses
w = clamp(w, w_min, w_max)
```

This is a simple element-wise tensor operation — trivially parallel on GPU.

#### 5.2 Three-Factor Rules

For reward-modulated learning:
```
eligibility += STDP_update           // fast: every spike
eligibility *= decay                 // fast: every timestep
dw = reward_signal * eligibility     // slow: when reward arrives
```

The `reward_signal` comes from the neuromodulatory system (e.g., dopamine concentration
from the modulatory overlay).

#### 5.3 Structural Learning

- Synaptogenesis/elimination rules modify `M_structural`
- Activity-dependent myelination modifies delay tensor `D`
- Dendritic growth rules modify multi-compartment neuron morphology

#### 5.4 Custom Rules

Users define custom plasticity by:
1. Subclassing `PlasticityRule` in C++ (full control)
2. Providing equations in DSL (compiles to CUDA via NVRTC)
3. Providing a raw CUDA kernel (expert users)

### 6. Neuromodulation & Global Effects

#### 6.1 Neuromodulatory Systems

Each system is a `Module` maintaining a concentration field that modulates the
`M_modulatory` mask overlay:

| System          | Transmitter      | Effect on M_modulatory          | Timescale         |
| --------------- | ---------------- | ------------------------------- | ----------------- |
| Dopaminergic    | Dopamine         | Gates plasticity, reward signal | 100ms - minutes   |
| Serotonergic    | Serotonin (5-HT) | Mood, impulse control           | seconds - hours   |
| Noradrenergic   | Norepinephrine   | Gain modulation, attention      | 100ms - seconds   |
| Cholinergic     | Acetylcholine    | Attention, memory encoding      | 100ms - seconds   |
| Histaminergic   | Histamine        | Arousal, wakefulness            | minutes           |
| Endocannabinoid | Endocannabinoids | Retrograde signaling, STD       | seconds           |
| Opioid          | Endorphins       | Pain modulation, reward         | seconds - minutes |
| Purinergic      | ATP, Adenosine   | Sleep pressure, glia signaling  | minutes - hours   |

#### 6.2 Volume Transmission

Neuromodulators act via diffusion, not point-to-point synapses:

```cpp
class VolumeTransmitter : public Module {
  Tensor concentration_field;    // per-region or per-voxel concentration
  float release_rate;
  float reuptake_rate;
  float degradation_rate;
  float diffusion_coefficient;

  void forward(State& state, Time t, Duration dt) override;
  Tensor sample_at(Tensor positions);  // query concentration at locations
};
```

The concentration field feeds into `M_modulatory`:
```
M_modulatory[block] = gain_function(concentration_field.sample_at(block_center))
```

#### 6.3 Systemic / Hormonal Effects

Global state modifiers implemented as `nrn::GlobalModulator`:

- **Stress (cortisol)**: modulates excitability, plasticity rates
- **Circadian rhythm**: oscillating parameters (arousal, plasticity windows)
- **Metabolic state**: energy availability affecting computation
- **Pharmacological agents**: alcohol (GABA↑, NMDA↓), caffeine (adenosine↓),
  SSRIs (serotonin reuptake↓), etc.
- **Inflammatory state**: cytokine levels affecting neural function
- **Hormonal cycles**: estrogen/testosterone effects on excitability

Each is a function that modifies model parameters or mask values globally:
```
effective_params = base_params * Π(global_modifier_i(systemic_state))
```

### 7. Extracellular Environment

Often overlooked but important for accuracy:

- **Ion concentrations**: extracellular K+, Na+, Ca2+ dynamics (K+ accumulation during
  intense activity changes excitability — creates positive feedback loops)
- **Neurotransmitter spillover**: glutamate/GABA diffusion beyond synaptic cleft
  (feeds into astrocyte detection)
- **Field potentials**: LFP/EEG proxy computed from transmembrane currents (useful for
  validation against experimental recordings)
- **Ephaptic coupling**: electric field effects between closely packed neurons (adds
  another coupling term to state update)
- **pH dynamics**: activity-dependent pH changes affecting channel function

Implementation: coarse-grained diffusion solver on GPU. Can be per-region (fast, low
resolution) or per-voxel (slower, needed for accurate extracellular dynamics).

---

## Part IV: Timescale Architecture

### The Multi-Timescale Problem

| Process                 | Timescale        | Update Strategy            |
| ----------------------- | ---------------- | -------------------------- |
| Action potential        | ~1 ms            | Event-driven (spike queue) |
| Synaptic transmission   | 1-100 ms         | Clock (dt_fast = 0.1ms)    |
| Short-term plasticity   | 100 ms - 10 s    | Clock (dt_fast)            |
| Astrocyte calcium       | 1-30 s           | Clock (dt_slow = 1-100ms)  |
| Neuromodulation         | 100 ms - minutes | Clock (dt_slow)            |
| STDP / Hebbian learning | 10 ms - seconds  | Event-triggered + clock    |
| Structural plasticity   | minutes - hours  | Periodic (dt_structural)   |
| Myelination changes     | hours - days     | Periodic (dt_structural)   |
| Microglial remodeling   | hours - days     | Periodic (dt_structural)   |
| Hormonal / systemic     | minutes - hours  | Periodic (dt_global)       |
| Developmental           | days - weeks     | Periodic (dt_global)       |

### Hierarchical Multi-Rate with Hybrid Event/Clock

**Chosen approach: Option D — hybrid event/clock with lazy evaluation.**

```
┌─────────────────────────────────────────────────────────────┐
│ Level 0: Spike Events           (event-driven, sub-ms)      │
│   Sparse spike events batched into time bins for GPU        │
│   Parallel scan for linear membrane dynamics                │
│                                                             │
│ Level 1: Fast Clock             (dt = 0.1-1.0 ms)           │
│   Membrane dynamics, synaptic currents, STP                 │
│   All point neurons updated every step (dense parallel)     │
│   Spike detection + reset                                   │
│                                                             │
│ Level 2: Medium Clock           (dt = 1-100 ms)             │
│   Astrocyte dynamics, neuromodulation                       │
│   Lazy: only update if inputs changed significantly         │
│   Neuron→Astro coupling at level boundary                   │
│   Astro→Neuron/Synapse coupling at level boundary           │
│                                                             │
│ Level 3: Slow Clock             (dt = 1-60 s)               │
│   Plasticity consolidation, homeostatic scaling             │
│   Structural mask updates (synaptogenesis/elimination)      │
│                                                             │
│ Level 4: Ultra-Slow             (dt = minutes-hours)        │
│   Myelination, microglial remodeling                        │
│   Global hormonal/systemic updates                          │
│   Block structure repartitioning                            │
│   Checkpoint save                                           │
└─────────────────────────────────────────────────────────────┘
```

**Coupling between levels:**
- Fast→Slow: accumulate (spike counts, activity statistics) over fast window,
  deliver as summary at slow boundary
- Slow→Fast: schedule events (SICs) or update parameters (M_modulatory) at slow
  boundary, take effect during next fast window
- This is **operator splitting** (Strang or Lie-Trotter)

**Lazy evaluation for slow processes:**
- Astrocytes that haven't received significant input skip their update
- "Significant" = threshold on accumulated input (configurable)
- Tracks a dirty flag per astrocyte; only dirty astrocytes update
- On GPU: compact dirty astrocytes into a dense sub-array, update only those
- Falls back to full update if >50% are dirty (better to just do all of them)

### Simulation Clock

```cpp
class Clock {
  Time global_time;                    // high-precision simulation time
  Duration dt_fast;                    // finest resolution (0.1 ms)
  Duration dt_slow;                    // medium resolution (1-100 ms)
  Duration dt_structural;             // slow resolution (seconds-minutes)
  Duration dt_global;                  // ultra-slow (minutes-hours)

  uint64_t fast_step = 0;
  uint64_t slow_ratio;                 // dt_slow / dt_fast
  uint64_t structural_ratio;           // dt_structural / dt_slow

  EventQueue<SpikeEvent> spike_queue;  // priority queue for spike events
  EventQueue<SICEvent> sic_queue;      // priority queue for SIC events

  void advance_fast();                 // one fast step
  void advance_slow();                 // one slow step (every slow_ratio fast steps)
  void advance_structural();           // one structural step
  bool is_slow_boundary() const;
  bool is_structural_boundary() const;
};
```

---

## Part V: Distributed Architecture

### Hardware Target

| Config        | Hardware                         | Target Scale                   |
| ------------- | -------------------------------- | ------------------------------ |
| Workstation   | 1× RTX 4090 (24 GB)              | 10^6 neurons + 10^5 astrocytes |
| Single DGX    | 8× A100/H100 (80 GB, NVSwitch)   | 10^7 neurons + 10^6 astrocytes |
| Small cluster | 8-32 DGX nodes (64-256 GPUs, IB) | 10^8 neurons + 10^7 astrocytes |
| Supercluster  | 256-1500 DGX (2K-12K GPUs)       | 10^9 neurons + 10^8 astrocytes |

### Communication Stack

```
Application:     nrn::Distributed (unified API)
                       |
                 ┌─────┴──────┐
                 │            │
Collectives:   NCCL        NVSHMEM
(global state  (allreduce,  (GPU-initiated one-sided
 sync)         broadcast)   puts for spike delivery)
                 │            │
                 └─────┬──────┘
                       │
Transport:       NVLink (intra-node) / InfiniBand (inter-node)
```

**Key insight from the research report**: All existing brain simulators use MPI with
CPU-mediated communication (GPU→CPU→MPI→CPU→GPU). libnrn uses NVSHMEM for GPU-initiated
spike delivery, eliminating CPU involvement entirely.

### Spike Communication via NVSHMEM

```
1. Source GPU: neuron fires → write (neuron_id, time) to local outgoing buffer
2. Source GPU: at batch boundary, for each target GPU:
     nvshmem_put(target_gpu_incoming_buffer + offset, local_outgoing_data, size, target_pe)
   This happens FROM WITHIN THE CUDA KERNEL — no CPU involved
3. Target GPU: process incoming spikes on next timestep
4. Overlap: spike delivery for step T runs concurrently with neuron update for step T+1
```

Spike compression options:
- Bitmap: for high-firing-rate populations (>5%), dense binary mask per timestep
- Sparse: for low-firing-rate (neuron_id, time pairs), delta-encoded
- Hybrid: per-block decision based on local firing rate

### Global State Synchronization

For neuromodulatory state, rewards, global statistics:
- NCCL allreduce for aggregated quantities (total firing rates, reward)
- NCCL broadcast for global parameter updates
- Infrequent: slow-changing state only syncs at slow clock boundaries

### Partitioning

Hierarchical spatial decomposition:

1. **Coarse**: divide 3D brain volume into regions → GPU groups
2. **Medium**: within region, partition into astrocyte-territory-aligned blocks → GPUs
   (astrocyte territories must NOT be split across GPUs)
3. **Fine**: within GPU, partition neurons into thread-block-sized groups

Block-dense connectivity means partition quality directly affects efficiency:
- Good partition: most dense blocks are GPU-local, few cross-GPU blocks
- Bad partition: many cross-GPU blocks, high communication

Use METIS/ParMETIS for graph partitioning. Repartition on ultra-slow timescale when
structural plasticity significantly changes the connectivity graph.

### Memory Budget (per GPU, 80 GB A100)

| Component                     | Per Unit    | Count/GPU     | Total      |
| ----------------------------- | ----------- | ------------- | ---------- |
| Point neuron state (AdEx)     | 32 B        | 1,000,000     | 32 MB      |
| Point astrocyte state (G-ChI) | 24 B        | 100,000       | 2.4 MB     |
| Block-dense weights           | 4 B/entry   | ~10^9 entries | ~4 GB      |
| Structural mask (packed bits) | 1 bit/entry | ~10^9 entries | ~125 MB    |
| Modulatory mask               | 2 B/entry   | ~10^9 entries | ~2 GB      |
| Delay tensor                  | 1 B/entry   | ~10^9 entries | ~1 GB      |
| Plasticity traces             | 8 B/syn     | ~10^9         | ~8 GB      |
| Spike buffers                 | —           | —             | ~2 GB      |
| NVSHMEM heap                  | —           | —             | ~4 GB      |
| Ring buffer (delays)          | —           | —             | ~2 GB      |
| Recording buffers             | —           | —             | ~2 GB      |
| **Total**                     |             |               | **~26 GB** |

~54 GB headroom for HH neurons, larger blocks, recording, or procedural connectivity
cache.

---

## Part VI: CUDA Kernel Architecture

### Design Principles

1. **Structure of Arrays (SoA)**: neuron state as separate contiguous arrays per variable.
   `v[N], w[N], I_syn[N]` — not `struct Neuron { v, w, I_syn } neurons[N]`.
   Enables coalesced memory access.

2. **Persistent kernels**: main simulation loop runs as a single long-running kernel
   that doesn't exit between timesteps. Avoids launch overhead.

3. **Tensor core utilization**: block-dense connectivity matmul uses tensor cores
   (wmma intrinsics or cuBLAS). This is where the masked-dense paradigm pays off —
   tensor cores are 16× faster than CUDA cores for matmul.

4. **Warp-level parallelism**: one warp per neuron group for local reductions.
   Cooperative groups for flexible sync.

5. **Shared memory**: cache frequently accessed blocks (current input weights,
   spike status of nearby neurons) in SMEM.

6. **Double buffering**: current state and next state in separate buffers to avoid
   race conditions between read and write.

7. **Kernel fusion**: fuse membrane_update + spike_detect + reset into single kernel.
   Fuse synaptic_current + membrane_update. Fuse plasticity_update + spike_processing.

### Memory Access Pattern

The main simulation step:

```
// Phase 1: Synaptic current delivery (the "matmul")
for each block (block_i, block_j) in block_map:
    // Load spike status of source neurons in block_j
    spike_vec = spikes[block_j * B : (block_j+1) * B]

    // Load block data (weight, masks)
    W = weights[block_idx]
    M = structural_mask[block_idx] & modulatory_mask[block_idx]

    // Block matmul: I_syn[block_i] += (W ⊙ M) @ spike_vec
    // This is a masked dense matmul — perfect for tensor cores

// Phase 2: Neuron state update (pointwise parallel)
for each neuron n in parallel:
    v[n] += dt * ((v_rest[n] - v[n]) / tau[n] + I_syn[n] / C[n])
    if v[n] > v_thresh[n]:
        emit_spike(n, t)
        v[n] = v_reset[n]

// Phase 3: Plasticity update (on connectivity tensors)
for each block with active pre or post spikes:
    update_traces(block)
    update_weights(block)  // tensor operation on W, masked by M
```

### Kernel Variants

| Neuron Model      | Kernel Strategy                                      |
| ----------------- | ---------------------------------------------------- |
| LIF               | Parallel scan (linear dynamics), pointwise threshold |
| AdEx/Izhikevich   | Forward Euler or exponential Euler, 2-var update     |
| HH                | RK2 or exponential Euler, 4-var update               |
| Multi-compartment | Hines tree-parallel per neuron, data-parallel across |

---

## Part VII: Configuration & DSL

### Model Definition (Three Entry Points)

**1. C++ API** (primary, performance-critical paths):
```cpp
auto pop = nrn::neuron::AdEx(N, nrn::AdExOptions().v_rest(-65.0_mV)...);
```

**2. Config files** (experiment specification, YAML):
```yaml
simulation:
  duration: 10.0s
  dt_fast: 0.1ms
  dt_slow: 10ms

populations:
  - name: excitatory
    model: AdEx
    count: 80000
    params:
      v_rest: -70.6mV
      v_thresh: -50.4mV
      # ...

connections:
  - source: excitatory
    target: excitatory
    topology: distance_dependent
    sigma: 200um
    synapse: AMPA
    plasticity: stdp
    block_size: 512
    representation: block_dense

astrocytes:
  - name: cortical_astro
    model: GChI
    count: 15000
    territory: voronoi
    covers: [exc_to_exc]  # which connections this astrocyte population covers
```

**3. Python bindings** (pybind11, stretch goal Phase 3):
```python
import libnrn as nrn
pop = nrn.neuron.AdEx(80000, v_rest=-70.6, v_thresh=-50.4)
```

**4. Equation DSL** (custom models, compiles to CUDA via NVRTC):
```
model MyNeuron:
  state: v (mV), w (nA)
  params: C_m (pF), g_L (nS), E_L (mV), V_T (mV), Delta_T (mV), a (nS), b (nA)
  dynamics:
    dv/dt = (-g_L * (v - E_L) + g_L * Delta_T * exp((v - V_T) / Delta_T) - w + I_syn) / C_m
    dw/dt = (a * (v - E_L) - w) / tau_w
  threshold: v > 0mV
  reset:
    v = E_L
    w += b
```

---

## Part VIII: Monitoring & I/O

### Recording

- **Spike trains**: sparse `(neuron_id, time)` pairs in ring buffer
- **State variables**: configurable subset, configurable subsample rate
- **Population statistics**: mean firing rate, CV_ISI, synchrony index, oscillations
- **Connectivity snapshots**: weight tensors at checkpoints
- **Neuromodulatory state**: concentration fields over time
- **LFP proxy**: computed from transmembrane currents

### Output Formats

- HDF5 with parallel I/O (primary)
- Zarr for cloud-native access
- NWB (Neurodata Without Borders) for experimental interop
- Parquet for spike trains (columnar, compressible)

### Checkpointing

- Full state serialization (same pattern as `torch.save`/`torch.load`)
- Incremental checkpoints for long runs
- Checkpoint-restart for fault tolerance
- Target: <5% overhead

### Real-Time Monitoring

- Streaming interface for live visualization (ZMQ or gRPC)
- Configurable probes (shouldn't impact performance)
- Event hooks for conditional recording

---

## Part IX: Validation & Testing

### Unit Validation

- Each neuron model vs analytical solutions / reference simulators
- Each synapse model vs known dynamics
- Each plasticity rule vs published results
- Astrocyte models vs De Pitta et al. reference traces

### Network Benchmarks

- Brunel balanced network (E-I): validated against NEST, analytical
- Tripartite synapse network: validated against Jiang et al. 2025 (NEST CPU)
- De Pitta & Brunel 2022 working memory: qualitative bistability reproduction

### Scaling Benchmarks

- Single GPU throughput: ≥10^9 neuron-timesteps/sec
- Weak scaling: ≥80% efficiency at 1024 GPUs
- Strong scaling: ≥60% at 256 GPUs for 10^8 neuron model

### Biological Plausibility

- Firing rate distributions match experimental ranges
- CV_ISI consistent with irregular firing
- Power spectral density of population activity
- Reproduce known experimental phenomena

---

## Part X: Technology Stack

| Component               | Technology                                |
| ----------------------- | ----------------------------------------- |
| Language                | C++20 (core) + CUDA 12+ (kernels)         |
| Build                   | CMake                                     |
| GPU compute             | CUDA, Cooperative Groups, Tensor Cores    |
| Multi-GPU (collectives) | NCCL                                      |
| Multi-GPU (p2p)         | NVSHMEM 3.0 + GPUDirect RDMA              |
| Transport               | NVLink 4.0 / NVSwitch + InfiniBand        |
| Partitioning            | METIS / ParMETIS                          |
| Math libraries          | cuBLAS, cuSPARSE, cuRAND                  |
| Serialization           | FlatBuffers or Protobuf                   |
| Config                  | yaml-cpp + nlohmann/json                  |
| Testing                 | GoogleTest + CUDA test utilities          |
| Python bindings         | pybind11                                  |
| Data I/O                | HDF5 (parallel), Zarr, Parquet            |
| Documentation           | Doxygen + Sphinx                          |
| DSL compilation         | NVRTC (runtime CUDA compilation)          |
| Containers              | Docker + NVIDIA NGC base                  |
| CI                      | GitHub Actions + NVIDIA Container Toolkit |

---

## Part XI: Project Phases

### Phase 1: Foundation (single-GPU spiking engine)
- Core tensor/state management on GPU (SoA layout)
- Module system and composition
- LIF, AdEx, Izhikevich neuron kernels
- Block-dense connectivity with mask overlays
- Static, exponential, double-exponential, AMPA synapses
- Topology generators: random, distance-dependent
- STDP plasticity (pair-based)
- Spike detection, ring buffer delays
- Basic monitoring and spike recording
- Config file loading (YAML)
- Benchmark: 10^6 neurons, 10^9 synapses, 1 GPU
- Validation: Brunel network against NEST

### Phase 2: Biological Depth (glia + neuromodulation, single-GPU)
- G-ChI and Li-Rinzel astrocyte models
- Tripartite synapse with territory assignment
- Multi-timescale orchestrator (operator splitting)
- Astrocyte gap junction coupling
- Gliotransmission → M_modulatory updates
- SIC emission and delivery
- Hodgkin-Huxley neuron model
- Full synapse library (NMDA, GABA_A, GABA_B, gap junctions)
- Short-term plasticity (Tsodyks-Markram)
- STDP variants (triplet, R-STDP), BCM, homeostatic scaling
- Neuromodulatory systems (dopamine, acetylcholine)
- Structural plasticity (mask updates)
- Extracellular K+ dynamics
- Validation: reproduce Jiang et al. 2025, De Pitta & Brunel 2022

### Phase 3: Scale (multi-GPU, multi-node)
- NVSHMEM spike delivery (GPU-initiated puts)
- NCCL global state synchronization
- Spatial decomposition + astrocyte-territory-aligned partitioning
- Ghost neuron management
- Two-tier communication (fast spikes + slow astro state)
- Communication-computation overlap
- MPI fallback
- Dynamic repartitioning
- Checkpointing and fault tolerance
- Performance profiling and optimization
- Benchmark: 10^7 on 1 DGX, 10^8 on 32 DGX nodes
- Scaling efficiency validation

### Phase 4: Ecosystem (DSL, interop, extreme scale)
- Equation-based DSL with NVRTC compilation
- Python bindings (pybind11)
- Brian2 equation importer
- NeuroML / SONATA / connectome data importers
- SWC morphology import
- NWB / Elephant / Neo export
- Multi-compartment neuron support
- Oligodendrocyte and microglia models
- Procedural connectivity for extreme scale
- Graph compilation / kernel fusion optimization
- Benchmark: 10^9 neurons on 2K-12K GPUs
- Documentation, tutorials, examples

---

## Part XII: Open Questions for Discussion

### Architecture

1. **Block-dense + mask vs literal sparse**: The masked-dense paradigm is the core
   architectural bet. Should we prototype both approaches early and benchmark, or
   commit to block-dense from the start? What block sizes should we target?

2. **Parallel scan for temporal dynamics**: How much of the neuron state evolution can
   we parallelize in time using scan operations? LIF is linear (perfect for scan),
   but AdEx/HH are nonlinear. Is operator splitting (linear part via scan + nonlinear
   correction) worth the complexity?

3. **Procedural vs stored connectivity**: At what scale do we switch from stored
   block-dense to procedural (on-the-fly computed) connectivity? Should procedural be
   the default with stored as optimization for irregular patterns?

4. **Memory management**: Custom GPU allocator (arena/pool) or CUDA memory pools?
   We need efficient allocation for dynamic mask updates and structural plasticity.

5. **Graph compilation**: `torch.compile`-style ahead-of-time kernel fusion for the
   simulation graph — Phase 1 or later?

### Neuroscience Fidelity

6. **Point vs compartmental**: Same Module interface for both, or separate abstractions?
   Multi-compartment is fundamentally a PDE (cable equation) while point neurons are ODEs.

7. **Astrocyte territorial overlap**: Biology has ~5% overlap. Strict Voronoi (simpler
   GPU partitioning) or allow overlap (more accurate but complicates decomposition)?

8. **Minimum astrocyte model**: Li-Rinzel (2 ODE) vs G-ChI (3 ODE) as default? What's
   the minimum that captures astroengram dynamics?

9. **Stochasticity budget**: How much stochasticity to support? Channel noise, vesicle
   release probability, spontaneous firing — each costs compute. Which are essential
   for biological plausibility vs nice-to-have?

### Interoperability & Use Cases

10. **PyTorch interop**: Should libnrn tensors be convertible to/from torch tensors for
    hybrid bio-plausible + deep learning models? (e.g., spiking network as a differentiable
    layer in a deep learning model)

11. **Differentiability**: Should we support gradients through the simulation (like
    Jaxley/BrainPy) for parameter fitting? Or is forward-only simulation sufficient?

12. **Real-time mode**: Wall-clock-synchronized mode for robotics/BCI? Or purely
    offline simulation speed?

13. **Primary use case priority**: Research accuracy vs engineering performance vs
    neuro-AI hybrid? This shapes every tradeoff.

### Naming & Identity

14. **Project name**: libnrn? ASTRAL? Something else? The name should reflect the
    scope (not just neurons — full neural substrate including glia).

---

## References

### Existing Simulators
- **NEST GPU**: 3.37B neurons on 12K GPUs (Golosio et al.)
- **GeNN**: procedural connectivity, 4.13M neurons on 1 GPU (Knight & Nowotny)
- **CARLsim**: DA-STDP, homeostatic scaling, multi-GPU (Beyeler et al.)
- **Brian2CUDA**: equation-based, glia-capable, single GPU
- **NEST CPU**: first-class astrocyte support (Jiang et al. 2025)
- **Arbor**: multi-compartment, multi-GPU, STDP
- **CoreNEURON**: morphologically detailed, GPU-accelerated (Blue Brain)
- **BrainPy**: JAX-based, event-driven, multi-device
- **Jaxley**: differentiable biophysical simulation (Nature Methods 2025)

### Key Neuroscience
- Williamson et al. "Learning-associated astrocyte ensembles." Nature (2025)
- Dewa et al. "Astrocytic ensemble as multiday trace." Nature (2025)
- Kozachkov et al. "Building transformers from neurons and astrocytes." PNAS (2023)
- De Pitta & Brunel. "Multiple forms of working memory." PNAS (2022)
- Jiang et al. "Modeling neuron-astrocyte interactions." PLOS Comp Bio (2025)

### Design Inspiration
- **libtorch**: API patterns, module system, tensor abstractions
- **Mamba/S4**: parallel scan for recurrence on GPUs
- **Flash Attention**: block-sparse with dense blocks on tensor cores
- **Megatron-LM**: multi-GPU model parallelism patterns
