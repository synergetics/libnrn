#include <nrn/monitor/state_recorder.h>

#include <utility>

namespace nrn {

StateRecorder::StateRecorder(std::string pop_name,
                             std::vector<std::string> variables,
                             int64_t subsample)
    : pop_name_(std::move(pop_name)),
      variables_(std::move(variables)),
      subsample_(subsample) {
    TORCH_CHECK(subsample_ >= 1,
                "StateRecorder: subsample must be >= 1, got ", subsample_);
    TORCH_CHECK(!variables_.empty(),
                "StateRecorder: at least one variable name is required");

    // Pre-create empty history entries so get() never misses.
    for (const auto& var : variables_) {
        history_[var] = {};
    }
}

void StateRecorder::record(const State& state, Time t) {
    ++call_count_;
    if ((call_count_ % subsample_) != 0) {
        return;
    }

    for (const auto& var : variables_) {
        if (state.contains(var)) {
            // Clone to CPU so recording doesn't hold GPU memory and
            // is immune to in-place overwrites of the source tensor.
            history_[var].push_back(state.get(var).detach().cpu().clone());
        }
    }
    times_.push_back(static_cast<float>(t));
}

void StateRecorder::reset() {
    for (auto& [key, vec] : history_) {
        vec.clear();
    }
    times_.clear();
    call_count_ = 0;
}

torch::Tensor StateRecorder::get(const std::string& variable) const {
    auto it = history_.find(variable);
    TORCH_CHECK(it != history_.end(),
                "StateRecorder: variable '", variable,
                "' is not being recorded for population '", pop_name_, "'");

    const auto& snapshots = it->second;
    if (snapshots.empty()) {
        return torch::zeros({0, 0}, torch::kFloat32);
    }

    // Stack snapshots [N] into [T, N].
    return torch::stack(snapshots, /*dim=*/0);
}

torch::Tensor StateRecorder::get_times() const {
    if (times_.empty()) {
        return torch::zeros({0}, torch::kFloat32);
    }
    return torch::from_blob(
               const_cast<float*>(times_.data()),
               {static_cast<int64_t>(times_.size())},
               torch::kFloat32)
        .clone();
}

} // namespace nrn
