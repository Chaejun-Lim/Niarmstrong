#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "pipeline_shared.hpp"
#include "posix_shm.hpp"

int main(int argc, char** argv) {
    std::string ctrl_name = "/hybrid_control";
    std::string vision_name = "/hybrid_vision";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ctrl") == 0 && i + 1 < argc) {
            ctrl_name = argv[++i];
        } else if (std::strcmp(argv[i], "--vision") == 0 && i + 1 < argc) {
            vision_name = argv[++i];
        }
    }

    try {
        PosixSharedMemory<ControlSharedBlock> ctrl(ctrl_name, false);
        PosixSharedMemory<VisionSharedBlock> vis(vision_name, false);

        std::cout << "[shm_monitor] ctrl=" << ctrl_name << " vision=" << vision_name << "\n";

        while (true) {
            const auto* c = ctrl.get();
            const auto* v = vis.get();
            const auto state_seq = c->state_seq.load(std::memory_order_acquire);
            const auto frame_seq = v->seq.load(std::memory_order_acquire);

            std::cout
                << "state_seq=" << state_seq
                << " ctrl_dt_ms=" << c->telemetry.loop_dt_ms
                << " overruns=" << c->telemetry.overruns
                << " leader_rx_ns=" << c->telemetry.leader_rx_timestamp_ns
                << " follower_tx_ns=" << c->telemetry.follower_tx_timestamp_ns
                << " follower_rx_ns=" << c->telemetry.follower_rx_timestamp_ns
                << " leader_rx_period_ms=" << c->telemetry.leader_rx_period_ms
                << " follower_tx_period_ms=" << c->telemetry.follower_tx_period_ms
                << " vision_seq=" << frame_seq
                << " zed_ms=" << v->telemetry.zed_ms
                << " yolo_ms=" << v->telemetry.yolo_ms
                << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    } catch (const std::exception& e) {
        std::cerr << "[shm_monitor] error: " << e.what() << "\n";
        return 1;
    }
}
