# Laser Aim Compensation Implementation Plan

**Goal:** Add robust camera-to-laser parallax and fixed mounting-bias compensation without changing the gimbal USB frame.

## 1. Configuration contract
- Add failing tests for defaults, YAML fields, finite values, distance ordering, and reprojection limits.
- Implement validated compensation fields in `GimbalUsbConfig` and example YAML.

## 2. Target-angle and pose compensation
- Add failing tests for unchanged center-only output, post-sign biases, synthetic PnP XYZ compensation, and fallback cases.
- Extend the converter with optional ordered corners and board dimensions.
- Implement validated planar PnP and safe fallback.

## 3. Safety gate and CLI integration
- Test that only `FULL_BOARD` supplies corners to pose compensation.
- Pass board dimensions from `AppConfig` during converter construction.
- Print compensation state at startup and latest valid angles headlessly.

## 4. Runbook
- Document coordinate convention, XYZ measurement, local YAML, bias calibration, three-distance validation, startup command, and laser safety.

## 5. Verification and commit
- Compile, run focused tests, run complete tests, review once, and commit on the current feature branch without merging to `main`.
