# Jetson Low-Latency Local Preview Design

**Date:** 2026-07-18

## Goal

Add a low-memory, low-latency OpenCV preview on the Jetson that shows only the confirmed green target outline and target center, while preserving the existing camera-tuning dashboard.

## User-visible modes

1. `ev-camera-tuning --local-preview`: run the existing dashboard and a Jetson-local OpenCV window from the same camera and detector session. Camera parameters may still be changed through the dashboard and the local window immediately follows the restarted camera session.
2. `ev-camera-preview --config config/default.yaml`: run only camera acquisition, detection, and the OpenCV window. FastAPI, Uvicorn, dashboard storage, JPEG encoding, and HTTP streaming are not imported or started. Camera exposure, gain, automatic modes, acquisition FPS, pixel format, and buffer size come from the YAML `camera:` section.

## Rendering and latency

The local window consumes the latest sequence-matched detection frame retained by `CameraTuningService`; it never queues old frames. It draws only a green four-corner outline and green center marker when `target_valid` is true. It performs no JPEG encoding and no network transport. Output width and refresh limit are command-line options.

## Runtime structure

Common camera/detector/service construction moves to a web-independent runtime module. The local preview module owns OpenCV rendering and the GUI loop. The standalone command imports neither the dashboard app nor Uvicorn. The existing dashboard command keeps its current behavior unless `--local-preview` is supplied.

For combined mode, the main thread owns the OpenCV GUI loop, while a Uvicorn server runs in a background thread. The service lifecycle is owned once by the command instead of both the GUI and FastAPI lifespan.

## Resource use

Standalone mode disables the unused image-diagnostics worker. The camera acquisition worker and detector worker remain enabled. It uses a latest-only frame path and does not retain a display queue.

## Exit and errors

`Q` or `Esc` closes the local window. `Ctrl+C` also stops the program. All paths stop the service and close OpenCV windows. GUI startup failures return a nonzero status with a message that tells the operator to run from the Jetson desktop with a valid `DISPLAY`.

## Safety

The 405 nm laser remains hardware-always-on whenever powered. Neither preview mode changes laser behavior. The laser must be physically disconnected or reliably covered during visual debugging.