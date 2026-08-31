#include <array>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <filesystem>
#include <termios.h>
#include <unistd.h>
#include <vector>

#if defined(__linux__)
#include <sys/ioctl.h>
#if __has_include(<asm/termbits.h>)
// Avoid struct termios name collision with <termios.h>.
#define termios asm_termios
#include <asm/termbits.h>
#undef termios
#endif
#endif

#include "pipeline_shared.hpp"
#include "posix_shm.hpp"

namespace {
std::atomic<bool> g_running{true};

constexpr uint8_t kStartBytes[4] = {0x7F, 0xFF, 0xFF, 0xFF};
constexpr uint8_t kCmdPosRef = 0x01;
constexpr uint8_t kCmdFeedback = 0x02;
constexpr uint8_t kDlcPosRef = 26;
constexpr uint8_t kDlcFeedback = 14;
constexpr size_t kFrameLenPosRef = 34;
constexpr size_t kFrameLenFeedback = 22;
constexpr const char* kLeaderByIdPort = "/dev/serial/by-id/usb-Arduino_RaspberryPi_Pico_472803531C5F2641-if00";
constexpr const char* kFollowerByIdPort = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0";

uint64_t now_ns() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void handle_signal(int) {
    g_running.store(false, std::memory_order_relaxed);
}

uint16_t crc16_modbus(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint16_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 0x0001) {
                crc = static_cast<uint16_t>((crc >> 1) ^ 0xA001);
            } else {
                crc >>= 1;
            }
        }
    }
    // Firmware/Python interface expect CRC bytes as (Hi, Lo) where this bitwise
    // routine yields swapped order relative to the table-based implementation.
    return static_cast<uint16_t>((crc << 8) | (crc >> 8));
}

speed_t to_baud(int baudrate) {
    switch (baudrate) {
        case 9600:
            return B9600;
        case 19200:
            return B19200;
        case 38400:
            return B38400;
        case 57600:
            return B57600;
        case 76800:
#ifdef B76800
            return B76800;
#else
            throw std::runtime_error("B76800 is not supported on this platform");
#endif
        case 115200:
            return B115200;
        case 230400:
            return B230400;
        default:
            throw std::runtime_error("Unsupported baudrate: " + std::to_string(baudrate));
    }
}

bool set_custom_baud_linux(int fd, int baudrate) {
#if defined(__linux__) && defined(TCGETS2) && defined(TCSETS2) && defined(BOTHER) && defined(CBAUD)
    termios2 tio2{};
    if (ioctl(fd, TCGETS2, &tio2) != 0) {
        return false;
    }
    tio2.c_cflag &= ~CBAUD;
    tio2.c_cflag |= BOTHER;
    tio2.c_ispeed = static_cast<speed_t>(baudrate);
    tio2.c_ospeed = static_cast<speed_t>(baudrate);
    return ioctl(fd, TCSETS2, &tio2) == 0;
#else
    (void)fd;
    (void)baudrate;
    return false;
#endif
}

std::string resolve_serial_port(const std::string& requested_port, const char* preferred_by_id, const char* tty_prefix) {
    const std::filesystem::path preferred(preferred_by_id);
    if (preferred.empty() || !preferred.has_root_path() || !std::filesystem::exists(preferred)) {
        return requested_port;
    }
    if (requested_port.rfind(tty_prefix, 0) == 0) {
        return preferred.string();
    }
    return requested_port;
}

class SerialPort {
public:
    SerialPort(const std::string& port, int baudrate) : fd_(-1), port_(port) {
        fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open " + port + ": " + std::string(std::strerror(errno)));
        }

        termios tty{};
        if (tcgetattr(fd_, &tty) != 0) {
            close(fd_);
            throw std::runtime_error("tcgetattr failed on " + port + ": " + std::string(std::strerror(errno)));
        }

        cfmakeraw(&tty);
        tty.c_cflag |= CLOCAL | CREAD;
        tty.c_cflag &= ~CRTSCTS;
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 0;

        bool use_custom_baud = false;
        speed_t speed = B115200;
        if (baudrate == 76800) {
#ifdef B76800
            speed = B76800;
#else
            // Placeholder for tcsetattr; actual 76800 is applied with termios2 below.
            speed = B38400;
            use_custom_baud = true;
#endif
        } else {
            speed = to_baud(baudrate);
        }
        cfsetispeed(&tty, speed);
        cfsetospeed(&tty, speed);

        if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
            close(fd_);
            throw std::runtime_error("tcsetattr failed on " + port + ": " + std::string(std::strerror(errno)));
        }

        if (use_custom_baud && !set_custom_baud_linux(fd_, baudrate)) {
            close(fd_);
            throw std::runtime_error(
                "Requested baudrate 76800 but platform does not expose B76800 and termios2(BOTHER) setup failed"
            );
        }
    }

    ~SerialPort() {
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    ssize_t read_some(uint8_t* dst, size_t n) {
        return ::read(fd_, dst, n);
    }

    ssize_t write_all(const uint8_t* src, size_t n) {
        size_t total = 0;
        while (total < n) {
            const ssize_t w = ::write(fd_, src + total, n - total);
            if (w < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                return -1;
            }
            total += static_cast<size_t>(w);
        }
        return static_cast<ssize_t>(total);
    }

private:
    int fd_;
    std::string port_;
};

struct Calibration {
    std::array<float, kJointCount> leader_zero{};
    std::array<float, kJointCount> direction{};
    std::array<float, kJointCount> follower_zero{};
    std::array<float, kJointCount> minimum{};
    std::array<float, kJointCount> maximum{};
};

std::array<float, kJointCount> parse_csv7_text(const std::string& raw, const char* label) {
    std::array<float, kJointCount> out{};
    std::stringstream ss(raw);
    std::string token;
    int idx = 0;
    while (std::getline(ss, token, ',')) {
        if (idx >= kJointCount) {
            break;
        }
        out[idx++] = std::stof(token);
    }
    if (idx != kJointCount) {
        throw std::runtime_error(std::string("Expected 7 comma-separated values in ") + label);
    }
    return out;
}

std::array<float, kJointCount> parse_csv7(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw) {
        throw std::runtime_error(std::string("Missing env var: ") + name);
    }
    return parse_csv7_text(raw, name);
}

std::array<float, kJointCount> parse_json_float_array(const std::string& json_text, const char* key) {
    const auto key_pos = json_text.find(key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error(std::string("Missing JSON key: ") + key);
    }

    const auto array_begin = json_text.find('[', key_pos);
    if (array_begin == std::string::npos) {
        throw std::runtime_error(std::string("Malformed JSON array for key: ") + key);
    }

    std::array<float, kJointCount> out{};
    size_t idx = 0;
    size_t pos = array_begin + 1;
    while (idx < kJointCount && pos < json_text.size()) {
        while (pos < json_text.size() && (json_text[pos] == ' ' || json_text[pos] == '\n' || json_text[pos] == '\r' || json_text[pos] == '\t')) {
            ++pos;
        }
        if (pos >= json_text.size() || json_text[pos] == ']') {
            break;
        }

        size_t end = pos;
        while (end < json_text.size() && json_text[end] != ',' && json_text[end] != ']') {
            ++end;
        }

        const std::string token = json_text.substr(pos, end - pos);
        out[idx++] = std::stof(token);
        pos = end + 1;
    }

    if (idx != kJointCount) {
        throw std::runtime_error(std::string("Expected 7 values in JSON key: ") + key);
    }

    return out;
}

std::filesystem::path resolve_leader_zero_json_path() {
    const char* env_path = std::getenv("LEADER_ZERO_JSON");
    if (env_path && env_path[0] != '\0') {
        return std::filesystem::path(env_path);
    }

    std::array<std::filesystem::path, 4> candidates = {
        std::filesystem::current_path() / "../leader_zero.json",
        std::filesystem::current_path() / "leader_zero.json",
        std::filesystem::absolute(std::filesystem::current_path()).parent_path() / "leader_zero.json",
        std::filesystem::path("../leader_zero.json"),
    };

    for (const auto& p : candidates) {
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    return {};
}

Calibration read_calibration_from_env() {
    Calibration c;
    const auto json_path = resolve_leader_zero_json_path();
    if (json_path.empty()) {
        c.leader_zero = parse_csv7("LEADER_ZERO_DEG");
    } else {
        std::ifstream in(json_path);
        if (!in) {
            c.leader_zero = parse_csv7("LEADER_ZERO_DEG");
        } else {
            std::stringstream buffer;
            buffer << in.rdbuf();
            const auto& json_text = buffer.str();
            try {
                c.leader_zero = parse_json_float_array(json_text, "\"zero_offsets\"");
            } catch (...) {
                c.leader_zero = parse_csv7("LEADER_ZERO_DEG");
            }
        }
    }

    c.direction = parse_csv7("LEADER_DIRECTION");
    c.follower_zero = parse_csv7("FOLLOWER_ZERO_DEG");
    c.minimum = parse_csv7("FOLLOWER_MIN_DEG");
    c.maximum = parse_csv7("FOLLOWER_MAX_DEG");
    return c;
}

std::array<float, kJointCount> to_target(const Calibration& c, const std::array<float, kJointCount>& leader) {
    std::array<float, kJointCount> out{};
    for (int i = 0; i < kJointCount; ++i) {
        float v = (leader[i] - c.leader_zero[i]) * c.direction[i] + c.follower_zero[i];
        if (v < c.minimum[i]) v = c.minimum[i];
        if (v > c.maximum[i]) v = c.maximum[i];
        out[i] = v;
    }
    return out;
}

std::optional<std::array<float, kJointCount>> parse_leader_line(const std::string& line) {
    std::array<float, kJointCount> values{};
    std::stringstream ss(line);
    std::string token;
    int idx = 0;
    while (std::getline(ss, token, ',')) {
        if (idx >= kJointCount) {
            return std::nullopt;
        }
        try {
            values[idx++] = std::stof(token);
        } catch (...) {
            return std::nullopt;
        }
    }
    if (idx != kJointCount) {
        return std::nullopt;
    }
    return values;
}

class LeaderReader {
public:
    explicit LeaderReader(SerialPort& serial) : serial_(serial) {}

    std::optional<std::array<float, kJointCount>> read_latest() {
        uint8_t buf[512];
        while (true) {
            const ssize_t n = serial_.read_some(buf, sizeof(buf));
            if (n <= 0) {
                break;
            }
            rx_.append(reinterpret_cast<const char*>(buf), static_cast<size_t>(n));
        }

        size_t pos = rx_.rfind('\n');
        if (pos == std::string::npos) {
            return std::nullopt;
        }

        std::string complete = rx_.substr(0, pos + 1);
        rx_.erase(0, pos + 1);

        size_t last_start = complete.rfind('\n', complete.size() > 1 ? complete.size() - 2 : 0);
        std::string line = (last_start == std::string::npos) ? complete : complete.substr(last_start + 1);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty()) {
            return std::nullopt;
        }
        return parse_leader_line(line);
    }

private:
    SerialPort& serial_;
    std::string rx_;
};

class FollowerController {
public:
    explicit FollowerController(SerialPort& serial) : serial_(serial) {}

    bool send_action(const std::array<float, kJointCount>& action_deg) {
        std::array<uint8_t, kFrameLenPosRef> frame{};
        frame[0] = kStartBytes[0];
        frame[1] = kStartBytes[1];
        frame[2] = kStartBytes[2];
        frame[3] = kStartBytes[3];
        frame[4] = kCmdPosRef;
        frame[5] = kDlcPosRef;

        for (int i = 0; i < 6; ++i) {
            const double scaled = static_cast<double>(action_deg[i]) * 10000.0;
            const int32_t raw = static_cast<int32_t>(std::llround(scaled));
            const size_t off = static_cast<size_t>(6 + i * 4);
            frame[off + 0] = static_cast<uint8_t>((raw >> 24) & 0xFF);
            frame[off + 1] = static_cast<uint8_t>((raw >> 16) & 0xFF);
            frame[off + 2] = static_cast<uint8_t>((raw >> 8) & 0xFF);
            frame[off + 3] = static_cast<uint8_t>(raw & 0xFF);
        }

        const float gripper = std::max(0.0f, std::min(100.0f, action_deg[6]));
        const uint16_t gr_raw = static_cast<uint16_t>(std::lround((gripper / 100.0f) * 65535.0f));
        frame[30] = static_cast<uint8_t>((gr_raw >> 8) & 0xFF);
        frame[31] = static_cast<uint8_t>(gr_raw & 0xFF);

        const uint16_t crc = crc16_modbus(frame.data(), 32);
        frame[32] = static_cast<uint8_t>((crc >> 8) & 0xFF);
        frame[33] = static_cast<uint8_t>(crc & 0xFF);

        return serial_.write_all(frame.data(), frame.size()) == static_cast<ssize_t>(frame.size());
    }

    std::optional<std::array<float, kJointCount>> read_state_sample() {
        uint8_t buf[256];
        while (true) {
            const ssize_t n = serial_.read_some(buf, sizeof(buf));
            if (n <= 0) {
                break;
            }
            rx_.insert(rx_.end(), buf, buf + n);
        }

        while (rx_.size() >= kFrameLenFeedback) {
            size_t start = find_start();
            if (start == std::string::npos) {
                rx_.clear();
                return std::nullopt;
            }
            if (start > 0) {
                rx_.erase(rx_.begin(), rx_.begin() + static_cast<long>(start));
            }
            if (rx_.size() < kFrameLenFeedback) {
                return std::nullopt;
            }

            if (rx_[4] != kCmdFeedback || rx_[5] != kDlcFeedback) {
                rx_.erase(rx_.begin());
                continue;
            }

            const uint16_t crc_recv = static_cast<uint16_t>((rx_[20] << 8) | rx_[21]);
            const uint16_t crc_calc = crc16_modbus(rx_.data(), 20);
            if (crc_recv != crc_calc) {
                rx_.erase(rx_.begin());
                continue;
            }

            std::array<float, kJointCount> state{};
            for (int i = 0; i < 6; ++i) {
                const int16_t raw = static_cast<int16_t>((rx_[6 + i * 2] << 8) | rx_[7 + i * 2]);
                state[i] = static_cast<float>(raw) / 10.0f;
            }
            const uint16_t gr_raw = static_cast<uint16_t>((rx_[18] << 8) | rx_[19]);
            state[6] = (static_cast<float>(gr_raw) * 100.0f) / 65535.0f;
            rx_.erase(rx_.begin(), rx_.begin() + static_cast<long>(kFrameLenFeedback));
            return state;
        }

        return std::nullopt;
    }

private:
    size_t find_start() const {
        for (size_t i = 0; i + 3 < rx_.size(); ++i) {
            if (rx_[i] == kStartBytes[0] && rx_[i + 1] == kStartBytes[1] && rx_[i + 2] == kStartBytes[2] && rx_[i + 3] == kStartBytes[3]) {
                return i;
            }
        }
        return std::string::npos;
    }

    SerialPort& serial_;
    std::vector<uint8_t> rx_;
};

}  // namespace

int main(int argc, char** argv) {
    std::string shm_name = "/hybrid_control";
    std::string leader_port = "/dev/ttyACM0";
    std::string follower_port = "/dev/ttyUSB0";
    int leader_baud = 115200;
    int follower_baud = 76800;
    double control_hz = 100.0;
    bool accept_policy = false;
    bool policy_only = false;
    std::string policy_shm_name = "/hybrid_policy";
    double policy_timeout_s = 0.25;
    std::string policy_bootstrap_deg_csv;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shm") == 0 && i + 1 < argc) {
            shm_name = argv[++i];
        } else if (std::strcmp(argv[i], "--hz") == 0 && i + 1 < argc) {
            control_hz = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--leader-port") == 0 && i + 1 < argc) {
            leader_port = argv[++i];
        } else if (std::strcmp(argv[i], "--follower-port") == 0 && i + 1 < argc) {
            follower_port = argv[++i];
        } else if (std::strcmp(argv[i], "--leader-baud") == 0 && i + 1 < argc) {
            leader_baud = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--follower-baud") == 0 && i + 1 < argc) {
            follower_baud = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--accept-policy") == 0) {
            accept_policy = true;
        } else if (std::strcmp(argv[i], "--policy-only") == 0) {
            policy_only = true;
        } else if (std::strcmp(argv[i], "--policy-shm") == 0 && i + 1 < argc) {
            policy_shm_name = argv[++i];
        } else if (std::strcmp(argv[i], "--policy-timeout-s") == 0 && i + 1 < argc) {
            policy_timeout_s = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--policy-bootstrap-deg") == 0 && i + 1 < argc) {
            policy_bootstrap_deg_csv = argv[++i];
        }
    }

    if (control_hz <= 0.0 || policy_timeout_s <= 0.0) {
        std::cerr << "--hz and --policy-timeout-s must be positive\n";
        return 1;
    }
    if (policy_only && !accept_policy) {
        std::cerr << "--policy-only requires --accept-policy\n";
        return 1;
    }
    if (policy_only && policy_bootstrap_deg_csv.empty()) {
        std::cerr << "--policy-only requires --policy-bootstrap-deg with the follower's current 7-joint pose\n";
        return 1;
    }

    if (!policy_only) {
        leader_port = resolve_serial_port(leader_port, kLeaderByIdPort, "/dev/ttyACM");
    }
    follower_port = resolve_serial_port(follower_port, kFollowerByIdPort, "/dev/ttyUSB");

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try {
        std::unique_ptr<SerialPort> leader_serial;
        std::unique_ptr<LeaderReader> leader;
        Calibration calibration{};
        if (!policy_only) {
            calibration = read_calibration_from_env();
            leader_serial = std::make_unique<SerialPort>(leader_port, leader_baud);
            leader = std::make_unique<LeaderReader>(*leader_serial);
        }
        SerialPort follower_serial(follower_port, follower_baud);
        FollowerController follower(follower_serial);

        PosixSharedMemory<ControlSharedBlock> shm(shm_name, true);
        auto* block = shm.get();
        std::optional<PosixSharedMemory<PolicySharedBlock>> policy_shm;
        PolicySharedBlock* policy_block = nullptr;
        if (accept_policy) {
            policy_shm.emplace(policy_shm_name, true);
            policy_block = policy_shm->get();
            policy_block->cmd = ControlCommand{};
            policy_block->seq.store(0, std::memory_order_relaxed);
        }
        block->cmd = ControlCommand{};
        block->leader = ControlLeader{};
        block->state = ControlState{};
        block->telemetry = ControlTelemetry{};
        block->cmd_seq.store(0, std::memory_order_relaxed);
        block->leader_seq.store(0, std::memory_order_relaxed);
        block->state_seq.store(0, std::memory_order_relaxed);
        block->stop_flag.store(0, std::memory_order_relaxed);

        std::array<float, kJointCount> last_action{};
        std::array<float, kJointCount> last_leader{};
        std::array<float, kJointCount> last_state{};
        bool have_action = false;
        bool have_leader = false;
        bool have_state = false;
        // A follower may emit feedback only after receiving a position command.
        // Before the rollout can produce its first policy action it needs that
        // feedback for its initial observation, so keep the legacy leader path
        // alive only until policy publishes its first valid command.
        bool policy_has_taken_control = false;
        if (policy_only) {
            // This is a hold command chosen by the operator to match the
            // follower's current pose. It wakes firmware that only emits
            // feedback after receiving a position command, without a leader.
            last_action = parse_csv7_text(policy_bootstrap_deg_csv, "--policy-bootstrap-deg");
            have_action = true;
        }
        uint64_t leader_rx_timestamp_ns = 0;
        uint64_t follower_tx_timestamp_ns = 0;
        uint64_t follower_rx_timestamp_ns = 0;
        uint64_t prev_leader_rx_timestamp_ns = 0;
        uint64_t prev_follower_tx_timestamp_ns = 0;
        double leader_rx_period_ms = -1.0;
        double follower_tx_period_ms = -1.0;

        const auto period = std::chrono::duration<double>(1.0 / control_hz);
        auto next_tick = std::chrono::steady_clock::now();

        std::cout << "[control_service] shm=" << shm_name << " hz=" << control_hz
                  << " leader=" << (policy_only ? "disabled" : leader_port) << " follower=" << follower_port
                  << " policy=" << (accept_policy ? policy_shm_name : "disabled")
                  << " bootstrap=" << (policy_only ? "hold" : "leader") << "\n";

        while (g_running.load(std::memory_order_relaxed) && block->stop_flag.load(std::memory_order_relaxed) == 0) {
            const auto loop_start = std::chrono::steady_clock::now();

            if (leader) {
                if (auto leader_angles = leader->read_latest(); leader_angles.has_value()) {
                    last_leader = *leader_angles;
                    have_leader = true;
                    last_action = to_target(calibration, *leader_angles);
                    have_action = true;
                    leader_rx_timestamp_ns = now_ns();
                    if (prev_leader_rx_timestamp_ns > 0 && leader_rx_timestamp_ns > prev_leader_rx_timestamp_ns) {
                        leader_rx_period_ms = static_cast<double>(leader_rx_timestamp_ns - prev_leader_rx_timestamp_ns) / 1'000'000.0;
                    }
                    prev_leader_rx_timestamp_ns = leader_rx_timestamp_ns;
                }
            }

            // Before policy produces its first action, use the leader in
            // normal policy mode, or the explicit stationary hold target in
            // policy-only mode. Afterwards a timeout never falls back to it.
            bool should_send_action = have_action;
            if (accept_policy) {
                should_send_action = !policy_has_taken_control && have_action;
                const uint64_t seq = policy_block->seq.load(std::memory_order_acquire);
                const ControlCommand policy_cmd = policy_block->cmd;
                const uint64_t age_ns = now_ns() > policy_cmd.timestamp_ns ? now_ns() - policy_cmd.timestamp_ns : 0;
                if (seq > 0 && policy_cmd.valid && age_ns <= static_cast<uint64_t>(policy_timeout_s * 1e9)) {
                    for (int i = 0; i < kJointCount; ++i) {
                        last_action[i] = policy_cmd.action_deg[i];
                    }
                    have_action = true;
                    policy_has_taken_control = true;
                    should_send_action = true;
                }
            }

            if (should_send_action) {
                if (follower.send_action(last_action)) {
                    follower_tx_timestamp_ns = now_ns();
                    if (prev_follower_tx_timestamp_ns > 0 && follower_tx_timestamp_ns > prev_follower_tx_timestamp_ns) {
                        follower_tx_period_ms = static_cast<double>(follower_tx_timestamp_ns - prev_follower_tx_timestamp_ns) / 1'000'000.0;
                    }
                    prev_follower_tx_timestamp_ns = follower_tx_timestamp_ns;
                }
            }

            if (have_leader) {
                ControlLeader leader_sample{};
                leader_sample.seq = block->leader_seq.load(std::memory_order_relaxed) + 1;
                leader_sample.timestamp_ns = leader_rx_timestamp_ns;
                leader_sample.valid = 1;
                for (int i = 0; i < kJointCount; ++i) {
                    leader_sample.leader_deg[i] = last_leader[i];
                }
                block->leader = leader_sample;
                block->leader_seq.store(leader_sample.seq, std::memory_order_release);
            }

            if (auto state = follower.read_state_sample(); state.has_value()) {
                last_state = *state;
                have_state = true;
                follower_rx_timestamp_ns = now_ns();
            }

            if (have_action) {
                ControlCommand cmd{};
                cmd.seq = block->cmd_seq.load(std::memory_order_relaxed) + 1;
                cmd.timestamp_ns = now_ns();
                cmd.valid = 1;
                for (int i = 0; i < kJointCount; ++i) {
                    cmd.action_deg[i] = last_action[i];
                }
                block->cmd = cmd;
                block->cmd_seq.store(cmd.seq, std::memory_order_release);
            }

            if (have_state) {
                ControlState state{};
                state.seq = block->state_seq.load(std::memory_order_relaxed) + 1;
                state.timestamp_ns = now_ns();
                state.valid = 1;
                for (int i = 0; i < kJointCount; ++i) {
                    state.state_deg[i] = last_state[i];
                }
                block->state = state;
                block->state_seq.store(state.seq, std::memory_order_release);
            }

            const auto loop_end = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
            block->telemetry.loop_count += 1;
            block->telemetry.loop_dt_ms = static_cast<double>(elapsed.count()) / 1000.0;
            block->telemetry.leader_rx_timestamp_ns = leader_rx_timestamp_ns;
            block->telemetry.follower_tx_timestamp_ns = follower_tx_timestamp_ns;
            block->telemetry.follower_rx_timestamp_ns = follower_rx_timestamp_ns;
            block->telemetry.leader_rx_period_ms = leader_rx_period_ms;
            block->telemetry.follower_tx_period_ms = follower_tx_period_ms;

            next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            if (next_tick < std::chrono::steady_clock::now()) {
                block->telemetry.overruns += 1;
                next_tick = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
            }
            std::this_thread::sleep_until(next_tick);
        }

        std::cout << "[control_service] stopped\n";
    } catch (const std::exception& e) {
        std::cerr << "[control_service] error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
