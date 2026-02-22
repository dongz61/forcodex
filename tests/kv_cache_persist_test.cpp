#include "core/logger.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace powerserve;

static const char *TEST_FILE_PATH = "/data/local/tmp/ziqian/powerserve/powerserve_kv_cache_persist.bin";
static constexpr size_t kFileSizeMB = 128;
static constexpr size_t kWriteSizeMB = 64;
static constexpr int kIterations = 20;
static constexpr size_t kChunkBytes = 4 * 1024 * 1024;

static double now_ms() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count() / 1000.0;
}

static void print_stats(const char *name, std::vector<double> values_ms) {
    if (values_ms.empty()) {
        POWERSERVE_LOG_INFO("{}: no samples", name);
        return;
    }
    std::sort(values_ms.begin(), values_ms.end());

    double sum = 0.0;
    for (double v : values_ms) sum += v;
    const double mean = sum / values_ms.size();
    const double p50 = values_ms[values_ms.size() / 2];
    const double p95 = values_ms[(values_ms.size() * 95) / 100];
    const double min_v = values_ms.front();
    const double max_v = values_ms.back();

    POWERSERVE_LOG_INFO(
        "{}: n={} mean={:.3f} ms p50={:.3f} ms p95={:.3f} ms min={:.3f} ms max={:.3f} ms",
        name,
        values_ms.size(),
        mean,
        p50,
        p95,
        min_v,
        max_v
    );
}

static bool preallocate_file(int fd, int64_t size_bytes) {
    const int rc = posix_fallocate(fd, 0, size_bytes);
    if (rc != 0) {
        POWERSERVE_LOG_WARN("posix_fallocate failed ({}), fallback to ftruncate", rc);
    } else {
        return true;
    }
    return ftruncate(fd, size_bytes) == 0;
}

static bool sync_file(int fd) {
    return fdatasync(fd) == 0;
}

static bool pwrite_all(int fd, const uint8_t *data, size_t bytes, int64_t offset) {
    size_t done = 0;
    while (done < bytes) {
        const size_t cur = std::min(kChunkBytes, bytes - done);
        const ssize_t written = pwrite(fd, data + done, cur, offset + static_cast<int64_t>(done));
        if (written <= 0) {
            return false;
        }
        done += static_cast<size_t>(written);
    }
    return true;
}

int main() {
    const int64_t file_size = static_cast<int64_t>(kFileSizeMB) * 1024 * 1024;
    const size_t write_size = static_cast<size_t>(kWriteSizeMB) * 1024 * 1024;

    POWERSERVE_LOG_INFO("==== KV cache persist test ====");
    POWERSERVE_LOG_INFO("file: {}", TEST_FILE_PATH);
    POWERSERVE_LOG_INFO("file_size={} MB write_size={} MB iterations={}", kFileSizeMB, kWriteSizeMB, kIterations);

    const int flags = O_CREAT | O_RDWR | O_TRUNC;
    const mode_t mode = 0644;
    const int fd = open(TEST_FILE_PATH, flags, mode);
    if (fd < 0) {
        POWERSERVE_LOG_ERROR("open failed: {}", std::strerror(errno));
        return 1;
    }

    if (!preallocate_file(fd, file_size)) {
        POWERSERVE_LOG_ERROR("preallocate failed: {}", std::strerror(errno));
        close(fd);
        return 2;
    }

    std::vector<uint8_t> write_buf(write_size, 0);
    for (size_t i = 0; i < write_buf.size(); ++i) {
        write_buf[i] = static_cast<uint8_t>(i & 0xFF);
    }

    std::vector<double> write_ms;
    std::vector<double> sync_ms;
    write_ms.reserve(kIterations);
    sync_ms.reserve(kIterations);

    // Always write into the same fixed region [0, write_size).
    const int64_t write_offset = 0;
    for (int i = 0; i < kIterations; ++i) {
        const double t0 = now_ms();
        const bool write_ok = pwrite_all(fd, write_buf.data(), write_buf.size(), write_offset);
        const double t1 = now_ms();
        if (!write_ok) {
            POWERSERVE_LOG_ERROR("write failed at iter {}: {}", i, std::strerror(errno));
            close(fd);
            return 3;
        }

        const bool sync_ok = sync_file(fd);
        const double t2 = now_ms();
        if (!sync_ok) {
            POWERSERVE_LOG_ERROR("sync failed at iter {}: {}", i, std::strerror(errno));
            close(fd);
            return 4;
        }

        write_ms.push_back(t1 - t0);
        sync_ms.push_back(t2 - t1);
        POWERSERVE_LOG_INFO("iter {:02d}: write={:.3f} ms sync={:.3f} ms total={:.3f} ms", i, t1 - t0, t2 - t1, t2 - t0);
    }

    print_stats("write_only", write_ms);
    print_stats("sync_only", sync_ms);

    close(fd);
    return 0;
}
