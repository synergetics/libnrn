#include <nrn/monitor/recorder.h>

// Recorder is a pure virtual base class — no implementation needed.
// This translation unit exists so that the vtable is emitted in exactly
// one object file (the "first virtual" rule) and to satisfy the CMake
// source list.
