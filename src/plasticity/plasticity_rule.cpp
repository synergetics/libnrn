#include <nrn/plasticity/plasticity_rule.h>

// PlasticityRule is a pure-virtual abstract base class.
// No implementation is needed here — all methods are either
// pure-virtual or defaulted (the virtual destructor).
//
// This translation unit exists so the build system has a .cpp
// to compile for the plasticity_rule component, and to ensure
// the vtable is emitted in exactly one translation unit.
