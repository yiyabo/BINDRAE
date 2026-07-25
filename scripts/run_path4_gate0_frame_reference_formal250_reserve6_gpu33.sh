#!/usr/bin/env bash
# Formal reference-only eligibility check for the six reserve systems at 250 steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Freeze the proposed reference-preconditioning contract for this confirmation.
export SOURCE_TAG="path4_gate0_full_reserve7_gpu33_20260723_011300_v3"
export TAG="${TAG:-path4_gate0_frame_reference_formal250_reserve6_gpu33_$(date +%Y%m%d_%H%M%S)_v1}"
export SOURCE_ROOT="logs/stage2/path4_gate0/$SOURCE_TAG"
export OUTPUT_ROOT="logs/stage2/path4_gate0/$TAG"
export PREPARED_SYSTEMS_DIR="processed_data/md_transition/context_full500_endpointdedup_20260716_v2/systems"
export PREPARED_FORCE_THRESHOLD="1000000"
export DIAGNOSTIC_FRAME_FORCE_THRESHOLD="1000000"
export ITERATION_COUNTS="250"
export SAMPLE_IDS="2e2o-A-BGC-400 2qje-D-Z8T-2 4wq2-B-3SU-301 6hfx-A-DMU-201 6lr4-C-CLR-301 7mql-A-RIO-302"

exec bash "$SCRIPT_DIR/run_path4_gate0_frame_reference_audit_gpu33.sh"
