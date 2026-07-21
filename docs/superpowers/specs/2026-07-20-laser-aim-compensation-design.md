# Laser Aim Compensation Design

**Date:** 2026-07-20
**Branch:** `feature/classical-white-board-tracking`

## Goal

Compensate a 405 nm laser whose emitter is not coaxial with the Hikrobot camera while preserving the existing 26-byte USB protocol and fail-closed detection gate. The gimbal continues to receive yaw/pitch errors; `distance_m`, `fire`, and other reserved payload fields remain unchanged.

## Coordinate and Sign Conventions

Camera coordinates follow OpenCV: `+X` image right, `+Y` image down, and `+Z` camera forward. The three `laser_offset_*_mm` values describe the laser aperture relative to the camera optical center in this coordinate frame.

Translation compensation is calculated in camera coordinates first. Existing `yaw_sign` and `pitch_sign` are then applied exactly once. `laser_yaw_bias_deg` and `laser_pitch_bias_deg` are additive offsets in final outgoing gimbal-command coordinates, so field calibration changes the bias in the same direction as the required outgoing correction.

## Configuration

Add backward-compatible fields to `GimbalUsbConfig`: pose enable, XYZ offsets, outgoing yaw/pitch biases, accepted pose distance range, and maximum reprojection error. All compensation defaults to disabled/zero. Board width and height come from the main project config rather than being duplicated in USB YAML.

## Compensation Algorithm

1. Validate and undistort the detected center and compute the existing center-only angle.
2. When pose compensation is enabled and a real `FULL_BOARD` observation supplies four ordered corners, solve planar PnP using calibrated intrinsics and the configured physical board size.
3. Accept only finite, forward-facing poses within the configured distance range and reprojection error.
4. Use `tvec - laser_offset` as the vector from laser aperture to target center, then calculate camera-frame yaw/pitch with `atan2(X,Z)` and `atan2(Y,Z)`.
5. Apply axis signs once, then add fixed outgoing biases.

## Fallback and Safety

If full-board pose is absent or rejected, use center-only aiming plus fixed biases. Partial, ring-only, and predicted observations never generate or reuse depth. Pose failure alone does not invalidate a target. Invalid calibration/center/final angles remain fail-closed. The USB packet layout and CRC do not change. `fire=0` is not a physical interlock for the hardware-always-on laser.

## Observability

Startup output reports compensation mode, offsets, biases, board dimensions, and pose limits. Headless runtime output includes the latest valid outgoing yaw/pitch.

## Tests

Cover config validation, zero-compensation compatibility, bias/sign order, synthetic projected-board XYZ parallax, bad-pose fallback, safety-gate corner routing, CLI board dimensions, and diagnostics.
