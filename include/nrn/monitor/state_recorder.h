#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include <nrn/core/state.h>
#include <nrn/core/types.h>
#include <nrn/monitor/recorder.h>

namespace nrn {

/// Records continuous state variables over time.
///
/// On each call to `record()`, the recorder clones the requested state
/// tensors and appends them to per-variable histories.  The histories
/// can be retrieved as a stacked tensor of shape [T, N] via `get()`.
class StateRecorder : public Recorder {
public:
    /// @param pop_name   Name of the population being recorded.
    /// @param variables  Names of state variables to track (e.g. {"v", "w"}).
    /// @param subsample  Record only every `subsample`-th call.
    StateRecorder(std::string pop_name,
                  std::vector<std::string> variables,
                  int64_t subsample = 1);

    /// Clone tracked variables from `state` and store them.
    void record(const State& state, Time t) override;

    /// Clear all stored history.
    void reset() override;

    /// Population name.
    const std::string& population_name() const override { return pop_name_; }

    /// Retrieve the recorded history for one variable.
    ///
    /// @return Tensor of shape [T, N] (T = number of recorded samples,
    ///         N = number of neurons).
    torch::Tensor get(const std::string& variable) const;

    /// Return the recording timestamps as a 1-D float tensor [T].
    torch::Tensor get_times() const;

    /// Variable names being recorded.
    const std::vector<std::string>& variables() const { return variables_; }

private:
    std::string pop_name_;
    std::vector<std::string> variables_;
    int64_t subsample_;
    int64_t call_count_ = 0;

    /// variable_name -> list of per-timestep snapshots (each [N]).
    std::unordered_map<std::string, std::vector<torch::Tensor>> history_;

    /// Recorded timestamps.
    std::vector<float> times_;
};

} // namespace nrn
