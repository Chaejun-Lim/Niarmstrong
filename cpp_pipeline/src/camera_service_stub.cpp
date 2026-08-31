#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <algorithm>

#include <sl/Camera.hpp>

#include "pipeline_shared.hpp"
#include "posix_shm.hpp"

namespace {
std::atomic<bool> g_running{true};
constexpr int kOutputWidth = 640;
constexpr int kOutputHeight = 480;
constexpr float kDepthMinMm = 200.0f;
constexpr float kDepthMaxMm = 1000.0f;

uint64_t now_ns() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void handle_signal(int) {
    g_running.store(false, std::memory_order_relaxed);
}

void draw_rect_rgb(
    uint8_t* image,
    int width,
    int height,
    int x1,
    int y1,
    int x2,
    int y2,
    uint8_t red,
    uint8_t green,
    uint8_t blue
) {
    x1 = std::clamp(x1, 0, width - 1);
    x2 = std::clamp(x2, 0, width - 1);
    y1 = std::clamp(y1, 0, height - 1);
    y2 = std::clamp(y2, 0, height - 1);
    if (x2 <= x1 || y2 <= y1) return;
    for (int x = x1; x <= x2; ++x) {
        for (int dy = 0; dy < 3 && y1 + dy <= y2; ++dy) {
            auto* top = image + (static_cast<size_t>(y1 + dy) * width + x) * 3;
            auto* bottom = image + (static_cast<size_t>(y2 - dy) * width + x) * 3;
            top[0] = bottom[0] = red; top[1] = bottom[1] = green; top[2] = bottom[2] = blue;
        }
    }
    for (int y = y1; y <= y2; ++y) {
        for (int dx = 0; dx < 3 && x1 + dx <= x2; ++dx) {
            auto* left = image + (static_cast<size_t>(y) * width + x1 + dx) * 3;
            auto* right = image + (static_cast<size_t>(y) * width + x2 - dx) * 3;
            left[0] = right[0] = red; left[1] = right[1] = green; left[2] = right[2] = blue;
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::string shm_name = "/hybrid_vision";
    double fps = 30.0;
    std::string depth_mode = "neural";
    double retrieve_scale = 1.0;
    std::string yolo_onnx;
    float yolo_conf = 0.25f;

    auto parse_depth_mode = [](const std::string& mode) {
        if (mode == "none") return sl::DEPTH_MODE::NONE;
        if (mode == "performance") return sl::DEPTH_MODE::PERFORMANCE;
        if (mode == "quality") return sl::DEPTH_MODE::QUALITY;
        if (mode == "ultra") return sl::DEPTH_MODE::ULTRA;
        if (mode == "neural") return sl::DEPTH_MODE::NEURAL;
        if (mode == "neural_plus") return sl::DEPTH_MODE::NEURAL_PLUS;
        if (mode == "neural_light") return sl::DEPTH_MODE::NEURAL_LIGHT;
        throw std::runtime_error("Unsupported --depth-mode: " + mode);
    };

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shm") == 0 && i + 1 < argc) {
            shm_name = argv[++i];
        } else if (std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            fps = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--depth-mode") == 0 && i + 1 < argc) {
            depth_mode = argv[++i];
        } else if (std::strcmp(argv[i], "--retrieve-scale") == 0 && i + 1 < argc) {
            retrieve_scale = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--yolo-onnx") == 0 && i + 1 < argc) {
            yolo_onnx = argv[++i];
        } else if (std::strcmp(argv[i], "--yolo-conf") == 0 && i + 1 < argc) {
            yolo_conf = static_cast<float>(std::stod(argv[++i]));
        }
    }
    if (fps <= 0.0) {
        std::cerr << "--fps must be positive\n";
        return 1;
    }
    if (retrieve_scale <= 0.0 || retrieve_scale > 1.0) {
        std::cerr << "--retrieve-scale must be within (0, 1]\n";
        return 1;
    }
    if (yolo_conf < 0.0f || yolo_conf > 1.0f) {
        std::cerr << "--yolo-conf must be within [0, 1]\n";
        return 1;
    }
    if (!yolo_onnx.empty() && !std::filesystem::exists(yolo_onnx)) {
        std::cerr << "--yolo-onnx file not found: " << yolo_onnx << "\n";
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try {
        PosixSharedMemory<VisionSharedBlock> shm(shm_name, true);
        auto* block = shm.get();

        block->telemetry = VisionTelemetry{};
        block->seq.store(0, std::memory_order_relaxed);
        block->stop_flag.store(0, std::memory_order_relaxed);
        block->width = 0;
        block->height = 0;
        block->depth_width = 0;
        block->depth_height = 0;
        std::memset(block->rgb, 0, sizeof(block->rgb));
        std::memset(block->yolo_rgb, 0, sizeof(block->yolo_rgb));
        std::memset(block->depth_mm, 0, sizeof(block->depth_mm));

        sl::Camera zed;
        sl::InitParameters init;
        init.camera_resolution = sl::RESOLUTION::HD1200;
        init.camera_fps = static_cast<unsigned int>(fps);
        init.depth_mode = parse_depth_mode(depth_mode);
        init.coordinate_units = sl::UNIT::MILLIMETER;
        init.depth_minimum_distance = kDepthMinMm;
        init.depth_maximum_distance = kDepthMaxMm;

        const auto status = zed.open(init);
        if (status != sl::ERROR_CODE::SUCCESS) {
            throw std::runtime_error("Failed to open ZED: " + std::string(sl::toString(status).c_str()));
        }

        sl::RuntimeParameters runtime;
        sl::Mat left;
        sl::Mat depth;
        sl::Objects objects;

        bool yolo_enabled = false;
        sl::ObjectDetectionRuntimeParameters od_runtime;
        od_runtime.detection_confidence_threshold = static_cast<unsigned int>(yolo_conf * 100.0f);

        if (!yolo_onnx.empty()) {
            sl::ObjectDetectionParameters od_params;
            od_params.enable_tracking = false;
            od_params.enable_segmentation = false;
            od_params.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_YOLOLIKE_BOX_OBJECTS;
            od_params.custom_onnx_file = sl::String(yolo_onnx.c_str());

            const auto od_status = zed.enableObjectDetection(od_params);
            if (od_status != sl::ERROR_CODE::SUCCESS) {
                throw std::runtime_error("Failed to enable ZED YOLO object detection: " + std::string(sl::toString(od_status).c_str()));
            }
            yolo_enabled = true;
        }

        sl::Resolution retrieve_resolution(kOutputWidth, kOutputHeight);

        const auto camera_info = zed.getCameraInformation();
        const int native_width = static_cast<int>(retrieve_resolution.width);
        const int native_height = static_cast<int>(retrieve_resolution.height);
        if (native_width > kVisionMaxWidth || native_height > kVisionMaxHeight) {
            throw std::runtime_error("ZED resolution exceeds shared-memory frame capacity");
        }

        const auto period = std::chrono::duration<double>(1.0 / fps);
        auto next_tick = std::chrono::steady_clock::now();

        std::cout << "[camera_service_cpp] shm=" << shm_name << " fps=" << fps
                  << " depth_mode=" << depth_mode
                  << " depth_range=" << kDepthMinMm << "-" << kDepthMaxMm << "mm"
                  << " native=" << camera_info.camera_configuration.resolution.width
                  << "x" << camera_info.camera_configuration.resolution.height
                  << " output=" << kOutputWidth << "x" << kOutputHeight;
        if (yolo_enabled) {
            std::cout << " yolo_onnx=" << yolo_onnx << " yolo_conf=" << yolo_conf;
        }
        std::cout << "\n";

        while (g_running.load(std::memory_order_relaxed) && block->stop_flag.load(std::memory_order_relaxed) == 0) {
            const auto frame_start = std::chrono::steady_clock::now();

            if (zed.grab(runtime) != sl::ERROR_CODE::SUCCESS) {
                next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
                std::this_thread::sleep_until(next_tick);
                continue;
            }
            if (zed.retrieveImage(left, sl::VIEW::LEFT, sl::MEM::CPU, retrieve_resolution) != sl::ERROR_CODE::SUCCESS) {
                next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
                std::this_thread::sleep_until(next_tick);
                continue;
            }
            if (zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU, retrieve_resolution) != sl::ERROR_CODE::SUCCESS) {
                next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
                std::this_thread::sleep_until(next_tick);
                continue;
            }

            const auto zed_end = std::chrono::steady_clock::now();
            const float zed_ms = static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(zed_end - frame_start).count() / 1000.0);

            float yolo_ms = 0.0f;
            uint8_t yolo_busy = 0;
            if (yolo_enabled) {
                const auto yolo_start = std::chrono::steady_clock::now();
                const auto od_ret = zed.retrieveObjects(objects, od_runtime);
                const auto yolo_end = std::chrono::steady_clock::now();
                yolo_ms = static_cast<float>(
                    std::chrono::duration_cast<std::chrono::microseconds>(yolo_end - yolo_start).count() / 1000.0);
                if (od_ret != sl::ERROR_CODE::SUCCESS) {
                    yolo_ms = 0.0f;
                    yolo_busy = 0;
                } else {
                    yolo_busy = 1;
                }
            }

            const int image_width = static_cast<int>(left.getWidth());
            const int image_height = static_cast<int>(left.getHeight());
            const int depth_width = static_cast<int>(depth.getWidth());
            const int depth_height = static_cast<int>(depth.getHeight());
            if (image_width > kVisionMaxWidth || image_height > kVisionMaxHeight ||
                depth_width > kVisionMaxWidth || depth_height > kVisionMaxHeight) {
                throw std::runtime_error("Retrieved ZED frame exceeds shared-memory frame capacity");
            }
            const auto* left_data = left.getPtr<sl::uchar1>(sl::MEM::CPU);
            const auto* depth_data = depth.getPtr<sl::float1>(sl::MEM::CPU);
            const size_t left_step = left.getStepBytes(sl::MEM::CPU);
            const size_t depth_step = depth.getStep<sl::float1>(sl::MEM::CPU);
            if (left_data == nullptr || depth_data == nullptr) {
                continue;
            }

            const uint64_t next_frame_seq = (block->seq.load(std::memory_order_relaxed) & ~uint64_t{1}) + 2;
            block->seq.store(next_frame_seq - 1, std::memory_order_release);
            block->width = static_cast<uint32_t>(image_width);
            block->height = static_cast<uint32_t>(image_height);
            block->depth_width = static_cast<uint32_t>(depth_width);
            block->depth_height = static_cast<uint32_t>(depth_height);
            auto* rgb = block->rgb;
            auto* yolo_rgb = block->yolo_rgb;
            for (int y = 0; y < image_height; ++y) {
                for (int x = 0; x < image_width; ++x) {
                    const auto* bgra = left_data + static_cast<size_t>(y) * left_step + static_cast<size_t>(x) * 4;
                    auto* rgb_pixel = rgb + (static_cast<size_t>(y) * image_width + x) * 3;
                    rgb_pixel[0] = bgra[2]; rgb_pixel[1] = bgra[1]; rgb_pixel[2] = bgra[0];
                }
            }
            std::memcpy(yolo_rgb, rgb, static_cast<size_t>(image_width) * image_height * 3);
            if (yolo_enabled) {
                for (const auto& object : objects.object_list) {
                    const auto& first = object.bounding_box[0];
                    const auto& third = object.bounding_box[2];
                    draw_rect_rgb(
                        yolo_rgb, image_width, image_height,
                        static_cast<int>(first.x), static_cast<int>(first.y),
                        static_cast<int>(third.x), static_cast<int>(third.y),
                        255, 64, 32
                    );
                }
            }
            for (int y = 0; y < depth_height; ++y) {
                for (int x = 0; x < depth_width; ++x) {
                    const float value = depth_data[static_cast<size_t>(y) * depth_step + x];
                    const float bounded = (std::isfinite(value) && value >= kDepthMinMm && value <= kDepthMaxMm)
                        ? value
                        : 0.0f;
                    block->depth_mm[static_cast<size_t>(y) * depth_width + x] = static_cast<uint16_t>(bounded);
                }
            }

            const auto frame_end = std::chrono::steady_clock::now();
            const float pipeline_ms = static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start).count() / 1000.0);

            VisionTelemetry telemetry{};
            telemetry.frame_seq = next_frame_seq;
            telemetry.timestamp_ns = now_ns();
            telemetry.zed_ms = zed_ms;
            telemetry.usb_ms = 0.0f;
            telemetry.yolo_ms = yolo_ms;
            telemetry.pipeline_ms = pipeline_ms;
            telemetry.yolo_busy = yolo_busy;

            block->telemetry = telemetry;
            block->seq.store(next_frame_seq, std::memory_order_release);

            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            std::this_thread::sleep_until(next_tick);
        }

        if (yolo_enabled) {
            zed.disableObjectDetection();
        }

        const uint64_t final_seq = block->seq.load(std::memory_order_relaxed);
        if (final_seq % 2 != 0) {
            block->seq.store(final_seq + 1, std::memory_order_release);
        }
        zed.close();
        std::cout << "[camera_service_cpp] stopped\n";
    } catch (const std::exception& e) {
        std::cerr << "[camera_service_cpp] error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
