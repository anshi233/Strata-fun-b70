// include/strata/program/qsa_width_option.hpp - host-only CLI/env decision for
// the integrated QSA width-bound path.
//
// One decision is made at startup, before any attention call.  An explicit
// --qsa-width-bound 0|1 always wins; when the flag is absent the legacy
// environment variable STRATA_QSA_WIDTH_BOUND is still honoured for
// compatibility, and otherwise the library default is OFF.  This header does
// no device work and is not used from a kernel.
#pragma once

#include <cstring>
#include <string>

namespace strata::program {

struct QsaWidthChoice {
    int cli = -1;              ///< -1 absent, 0 explicit off, 1 explicit on
    bool enabled = false;      ///< resolved host decision
    const char* source = "default";  ///< cli-0, cli-1, env-0, env-1, env-ignored, default
};

/// Parse exactly "0" or "1".  Any other value is an error, never a default.
inline bool qsa_width_parse_cli(const char* text, int& value, std::string& error) {
    if (text == nullptr) {
        error = "--qsa-width-bound requires 0 or 1";
        return false;
    }
    if (std::strcmp(text, "0") == 0) {
        value = 0;
        return true;
    }
    if (std::strcmp(text, "1") == 0) {
        value = 1;
        return true;
    }
    error = std::string("--qsa-width-bound must be 0 or 1, got '") + text + "'";
    return false;
}

/// Resolve CLI > legacy env > library default OFF into one startup decision.
/// An invalid CLI value is an error; a malformed legacy env value is ignored
/// exactly as the earlier experimental switch treated anything but the byte "1".
inline bool qsa_width_resolve(int cli, const char* legacy_env, bool& enabled,
                              const char*& source, std::string& error) {
    if (cli == 0 || cli == 1) {
        enabled = cli == 1;
        source = cli == 1 ? "cli-1" : "cli-0";
        return true;
    }
    if (cli != -1) {
        error = "--qsa-width-bound internal value is not 0, 1 or absent";
        return false;
    }
    if (legacy_env != nullptr && legacy_env[0] == '1' && legacy_env[1] == '\0') {
        enabled = true;
        source = "env-1";
        return true;
    }
    enabled = false;
    if (legacy_env != nullptr && legacy_env[0] == '0' && legacy_env[1] == '\0') {
        source = "env-0";
    } else if (legacy_env != nullptr && legacy_env[0] != '\0') {
        source = "env-ignored";
    } else {
        source = "default";
    }
    return true;
}

}  // namespace strata::program
