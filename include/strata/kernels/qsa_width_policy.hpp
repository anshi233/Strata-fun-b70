// include/strata/kernels/qsa_width_policy.hpp - host-only policy for the opt-in
// QSA width-bounded launch (tested math; integrated opt-in entry).
//
// The engine's existing CHUNK is 64.  This header performs no device work and
// does not change CHUNK.  It reads the width from the real [n][kStepCount]
// step buffer with the field constants from qsa.hpp, and it validates every
// record of one prompt chunk once; the batch
// loop can then take the last width in O(1) because the validated sequence is
// non-decreasing.
//
//   n_chunks_cap    = ceil(cap / 64)        scratch slots / stride (unchanged)
//   n_chunks_launch = ceil(max_width / 64)  grid x, all chunks are live or sentinel
//   n_chunks_active = n_chunks_launch       batch-wide merge upper bound
//
// Only plain INT8 + RS2 + no LANECELL + no pre75 is supported.  The caller must
// reject an enabled unsupported contract on the host before any launch; this
// header does not contain a fallback path.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int64_t kQsaWidthChunkCells = 64;

/// Startup host decision.  The CLI/env precedence is resolved once in
/// generate.cpp before any attention call; the kernels only read this bool.
inline bool& qsa_width_bound_state() {
    static bool enabled = false;
    return enabled;
}
inline void qsa_width_bound_set(bool enabled) { qsa_width_bound_state() = enabled; }
inline bool qsa_width_bound_enabled() { return qsa_width_bound_state(); }

struct QsaWidthPlan {
    bool enabled = false;
    int64_t n_chunks_cap = 0;
    int64_t n_chunks_launch = 0;
    int64_t n_chunks_active = 0;
};

/// The only accepted hint is 1 <= max_width <= cap.  Every scratch slot keeps
/// the cap stride; only the launch and merge traversal are bounded.
inline QsaWidthPlan qsa_width_plan(int64_t cap, int64_t max_width) {
    QsaWidthPlan p;
    if (cap <= 0 || max_width <= 0 || max_width > cap) return p;
    const int64_t cap_chunks = (cap + kQsaWidthChunkCells - 1) / kQsaWidthChunkCells;
    const int64_t active = (max_width + kQsaWidthChunkCells - 1) / kQsaWidthChunkCells;
    if (cap_chunks <= 0 || active <= 0 || active > cap_chunks) return p;
    p.enabled = true;
    p.n_chunks_cap = cap_chunks;
    p.n_chunks_launch = active;
    p.n_chunks_active = active;
    return p;
}

/// Only the plain INT8, RS2, non-LANECELL, non-pre75 path is supported.
inline bool qsa_width_supported(int kv_mode, int rs, bool lane_cell, bool pre75) {
    return kv_mode == 1 && rs == 2 && !lane_cell && !pre75;
}

/// Validate the actual interleaved step records of one prompt chunk:
/// step[i] = {pos, n_kv, n_bid, width} with the field offsets from qsa.hpp.
/// `expected_pos0` is the chunk's real p0; every record must match
/// qsa_step_fill(p0 + i), so resume/nonzero-p0 needs no p0 guess and no
/// compact-array wrapper.  Returns false for null/empty/bad cap, any field
/// mismatch, width outside [1, cap], or a non-decreasing violation.
inline bool qsa_width_steps_ok(const int32_t* steps, int64_t n, int64_t cap, const QsaShapes& s,
                               int64_t expected_pos0, int64_t* max_width) {
    if (steps == nullptr || n <= 0 || cap <= 0 || expected_pos0 < 0) return false;
    if (s.idx_block <= 0 || s.idx_top_k <= 0) return false;
    int64_t mx = 0;
    int64_t prev = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t* r = steps + i * (int64_t) kStepCount;
        const int64_t pos = r[kStepPos];
        const int64_t n_kv = r[kStepNKv];
        const int64_t n_bid = r[kStepNBid];
        const int64_t w = r[kStepWidth];
        if (pos != expected_pos0 + i) return false;
        if (n_kv != pos + 1) return false;
        if (n_bid != n_kv / s.idx_block) return false;
        if (w < 1 || w > cap) return false;
        if (w != qsa_selection_width(n_kv, s)) return false;
        if (i > 0 && w < prev) return false;
        prev = w;
        mx = w;
    }
    if (max_width != nullptr) *max_width = mx;
    return true;
}

/// Precondition: qsa_width_steps_ok accepted [0, T) for this chunk, so widths
/// are non-decreasing.  The batch maximum is therefore its last row's width.
inline int64_t qsa_width_batch_max(const int32_t* steps, int64_t t0, int64_t n) {
    return steps[(t0 + n - 1) * (int64_t) kStepCount + kStepWidth];
}

}  // namespace strata::kernels
