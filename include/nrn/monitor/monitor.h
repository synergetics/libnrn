/// @file monitor.h
/// @brief Aggregate include for the monitoring/recording subsystem.
///
/// @details
/// Pulls in the Recorder ops-table handle, SpikeRecorderState (records
/// discrete spike events), and StateRecorderState (records continuous
/// state variables at every timestep or subsampled).
///
/// @see Recorder, SpikeRecorderState, StateRecorderState

#pragma once

#include <nrn/monitor/recorder.h>
#include <nrn/monitor/spike_recorder.h>
#include <nrn/monitor/state_recorder.h>
