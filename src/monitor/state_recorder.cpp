#include <nrn/monitor/state_recorder.h>

#include <utility>

namespace nrn {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

StateRecorderState* state_recorder_create(std::string pop_name,
                                          std::vector<std::string> variables,
                                          int64_t subsample) {
    TORCH_CHECK(subsample >= 1,
                "state_recorder_create: subsample must be >= 1, got ",
                subsample);
    TORCH_CHECK(!variables.empty(),
                "state_recorder_create: at least one variable name is required");

    auto* s = new StateRecorderState{};
    s->pop_name = std::move(pop_name);
    s->variables = std::move(variables);
    s->subsample = subsample;
    s->call_count = 0;

    // Pre-create empty history entries so state_recorder_get() never misses.
    for (const auto& var : s->variables) {
        s->history[var] = {};
    }
    return s;
}

void state_recorder_destroy(StateRecorderState* s) {
    delete s;
}

// ---------------------------------------------------------------------------
// Ops implementations (void* self -> StateRecorderState*)
// ---------------------------------------------------------------------------

void state_recorder_record(void* self, const State& state, double t) {
    auto* s = static_cast<StateRecorderState*>(self);

    ++s->call_count;
    if ((s->call_count % s->subsample) != 0) {
        return;
    }

    for (const auto& var : s->variables) {
        if (state_contains(state, var)) {
            // Clone to CPU so recording doesn't hold GPU memory and
            // is immune to in-place overwrites of the source tensor.
            s->history[var].push_back(
                state_get(state, var).detach().cpu().clone());
        }
    }
    s->times.push_back(static_cast<float>(t));
}

void state_recorder_reset(void* self) {
    auto* s = static_cast<StateRecorderState*>(self);
    for (auto& [key, vec] : s->history) {
        vec.clear();
    }
    s->times.clear();
    s->call_count = 0;
}

const char* state_recorder_population_name(void* self) {
    auto* s = static_cast<StateRecorderState*>(self);
    return s->pop_name.c_str();
}

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------

torch::Tensor state_recorder_get(const StateRecorderState* s,
                                 const std::string& variable) {
    auto it = s->history.find(variable);
    TORCH_CHECK(it != s->history.end(),
                "state_recorder_get: variable '", variable,
                "' is not being recorded for population '", s->pop_name, "'");

    const auto& snapshots = it->second;
    if (snapshots.empty()) {
        return torch::zeros({0, 0}, torch::kFloat32);
    }

    // Stack snapshots [N] into [T, N].
    return torch::stack(snapshots, /*dim=*/0);
}

torch::Tensor state_recorder_get_times(const StateRecorderState* s) {
    if (s->times.empty()) {
        return torch::zeros({0}, torch::kFloat32);
    }
    return torch::from_blob(
               const_cast<float*>(s->times.data()),
               {static_cast<int64_t>(s->times.size())},
               torch::kFloat32)
        .clone();
}

const std::vector<std::string>& state_recorder_variables(
    const StateRecorderState* s) {
    return s->variables;
}

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

recorder_ops state_recorder_ops = {
    state_recorder_record,
    state_recorder_reset,
    state_recorder_population_name,
};

} // namespace nrn
