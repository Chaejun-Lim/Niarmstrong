#pragma once

#include <atomic>
#include <cstdint>

constexpr int kJointCount = 7;
constexpr int kVisionMaxWidth = 1920;
constexpr int kVisionMaxHeight = 1200;
constexpr size_t kVisionMaxPixels = static_cast<size_t>(kVisionMaxWidth) * kVisionMaxHeight;

struct ControlCommand {
    uint64_t seq;
    uint64_t timestamp_ns;
    float action_deg[kJointCount];
    uint8_t valid;
    uint8_t _pad[7];
};

struct ControlLeader {
    uint64_t seq;
    uint64_t timestamp_ns;
    float leader_deg[kJointCount];
    uint8_t valid;
    uint8_t _pad[7];
};

struct ControlState {
    uint64_t seq;
    uint64_t timestamp_ns;
    float state_deg[kJointCount];
    uint8_t valid;
    uint8_t _pad[7];
};

struct ControlTelemetry {
    uint64_t loop_count;
    double loop_dt_ms;
    uint64_t overruns;
    uint64_t leader_rx_timestamp_ns;
    uint64_t follower_tx_timestamp_ns;
    uint64_t follower_rx_timestamp_ns;
    double leader_rx_period_ms;
    double follower_tx_period_ms;
};

// A separate channel for autonomous policy actions.  It deliberately lives in
// its own POSIX SHM mapping so adding policy rollout support does not change
// the long-running teleoperation control mapping ABI.
struct PolicySharedBlock {
    std::atomic<uint64_t> seq;
    ControlCommand cmd;
};

struct ControlSharedBlock {
    std::atomic<uint64_t> cmd_seq;
    std::atomic<uint64_t> leader_seq;
    std::atomic<uint64_t> state_seq;
    std::atomic<uint8_t> stop_flag;
    uint8_t _pad0[7];

    ControlCommand cmd;
    ControlLeader leader;
    ControlState state;
    ControlTelemetry telemetry;
};

struct VisionTelemetry {
    uint64_t frame_seq;
    uint64_t timestamp_ns;
    float zed_ms;
    float usb_ms;
    float yolo_ms;
    float pipeline_ms;
    uint8_t yolo_busy;
    uint8_t _pad[3];
};

struct VisionSharedBlock {
    std::atomic<uint64_t> seq;
    std::atomic<uint8_t> stop_flag;
    uint8_t _pad0[7];
    VisionTelemetry telemetry;
    uint32_t width;
    uint32_t height;
    uint32_t depth_width;
    uint32_t depth_height;
    uint8_t rgb[kVisionMaxPixels * 3];
    uint8_t yolo_rgb[kVisionMaxPixels * 3];
    uint16_t depth_mm[kVisionMaxPixels];
};
