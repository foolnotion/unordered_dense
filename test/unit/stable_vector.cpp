#include <ankerl/unordered_dense.h>

#include <third-party/nanobench.h> // for Rng, doNotOptimizeAway, Bench

#include <doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#if _MSC_VER
#    include <intrin.h> // required for CLZ builtin
#endif

// constexpr version to find the index of the most significant bit. E.g. for 33 (100001b) this returns 5.
constexpr auto most_significant_active_bit_cx(size_t num) -> size_t {
    auto result = size_t();
    while ((num >>= 1U) != 0U) {
        ++result;
    }
    return result;
}

// Uses intrinsics to find the most significant bit. E.g. for 33 (100001b) this returns 5.
// On Windows this can't be constexpr, so we have a separate function for that.
inline auto most_significant_active_bit(size_t num) noexcept -> size_t {
#ifdef _MSC_VER
#    if UINTPTR_MAX == UINT32_MAX
    return 31U - __lzcnt(num);
#    else
    return 63U - __lzcnt64(num);
#    endif
#else
#    if UINTPTR_MAX == UINT32_MAX
    return 31U - static_cast<size_t>(__builtin_clz(num));
#    else
    return 63U - static_cast<size_t>(__builtin_clzll(num));
#    endif
#endif
}

// very simple random access container where increasing size does not invalidate any references
// StartSize = 8 == 0b1000
//   0b000 to   0b111 get into block[0] (size 8)
//  0b1000 to  0b1111 get into block[1] (size 8)
// 0b10000 to 0b11111 get into block[1] (size 16)

template <typename T, size_t MaxCapacity, typename Allocator = std::allocator<T>>
class stable_vector {
    static constexpr auto num_blocks = most_significant_active_bit_cx(MaxCapacity) - 1;

    T* m_first{};
    std::array<T*, num_blocks> m_blocks{};
    size_t m_size = 0;
    Allocator m_alloc{};

    static auto calc_block_idx(size_t i) noexcept -> std::size_t {
        return most_significant_active_bit(i);
    }

public:
    [[nodiscard]] auto size() const noexcept -> size_t {
        return m_size;
    }

    void grow() {
        if (0 == m_size) {
            m_first = m_alloc.allocate(1);
            m_size = 1;
        } else {
            m_blocks[calc_block_idx(m_size)] = m_alloc.allocate(m_size);
            m_size *= 2;
        }
    }

    ~stable_vector() {
        if (0 == m_size) {
            return;
        }
        m_alloc.deallocate(m_first, 1);
        auto n = calc_block_idx(m_size);
        auto block_size = size_t{1};
        for (std::size_t i = 0; i < n; ++i) {
            m_alloc.deallocate(m_blocks[i], block_size);
            block_size *= 2;
        }
    }

    constexpr auto operator[](size_t i) const noexcept -> T const& {
        if (i == 0) {
            return *m_first;
        }
        auto block_nr = most_significant_active_bit(i);
        auto mask = (size_t{1} << block_nr) - 1;
        return m_blocks[block_nr][i & mask];
    }

    auto operator[](size_t i) noexcept -> T& {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        return const_cast<T&>(std::as_const(*this)[i]);
    }
};

constexpr auto num_bits_closest(size_t max_val, size_t x) -> size_t {
    auto f = size_t{0};
    while (x << (f + 1) <= max_val) {
        ++f;
    }
    return f;
}

// behaves like a slot_map, with stable references.
// * O(1) construct(): Constructs an element, potentially allocating memory. Returns an index to the element. Might reuse
// previously allocated memory.
// * O(1) destroy(idx): destroys the element and puts it into an in-place freelist
// * O(1) operator[](idx): Accesses element at that index.
// * O(1) capacity(): total memory available

// Things it does NOT do: anything that passes over all constructed elements, like:
// * destruct all constructed elements. You need to do this yourself!
// * Iterate all elements. Needs outer storage
template <typename T, size_t BlockSizeBytes = 4096, typename Allocator = std::allocator<T>>
class stable_index_map {
    static constexpr auto num_bits = num_bits_closest(BlockSizeBytes, sizeof(T));
    static constexpr auto mask = (1U << num_bits) - 1U;
    std::vector<T*> m_blocks{};

public:
    // TODO(martinus) not yet implemented
    stable_index_map(stable_index_map const&) = delete;
    stable_index_map(stable_index_map&&) = delete;
    auto operator=(stable_index_map const&) -> stable_index_map& = delete;
    auto operator=(stable_index_map&&) -> stable_index_map& = delete;

    stable_index_map() = default;

    ~stable_index_map() {
        auto alloc = Allocator{};
        for (auto* ptr : m_blocks) {
            alloc.deallocate(ptr, 1U << num_bits);
        }
    }

    [[nodiscard]] constexpr auto size() const -> size_t {
        return m_blocks.size() << num_bits;
    }

    constexpr void grow() {
        auto alloc = Allocator{};
        m_blocks.push_back(alloc.allocate(1U << num_bits));
    }

    [[nodiscard]] constexpr auto operator[](size_t i) const noexcept -> T const& {
        return m_blocks[i >> num_bits][i & mask];
    }

    [[nodiscard]] constexpr auto operator[](size_t i) noexcept -> T& {
        return m_blocks[i >> num_bits][i & mask];
    }
};

TEST_CASE("stable_vector") {
    auto sv = stable_vector<size_t, std::numeric_limits<uint32_t>::max()>{};

    for (size_t i = 0; i < 4; ++i) {
        sv.grow();
    }
    auto capa = sv.size();
    for (size_t i = 0; i < capa; ++i) {
        sv[i] = i;
    }

    for (size_t i = 0; i < capa; ++i) {
        REQUIRE(sv[i] == i);
    }
}

TEST_CASE("stable_index_map") {
    auto sv = stable_index_map<size_t>{};

    for (size_t i = 0; i < 20; ++i) {
        sv.grow();
    }
    auto capa = sv.size();
    for (size_t i = 0; i < capa; ++i) {
        sv[i] = i;
    }

    for (size_t i = 0; i < capa; ++i) {
        REQUIRE(sv[i] == i);
    }
}

TEST_CASE("bench_stable_vector" * doctest::test_suite("bench") * doctest::skip()) {
    using namespace std::literals;

    ankerl::nanobench::Rng rng(123);
    auto sv = stable_vector<size_t, std::numeric_limits<uint32_t>::max()>{};
    for (size_t i = 0; i < 21; ++i) {
        sv.grow();
    }
    auto capa = sv.size();
    for (size_t i = 0; i < capa; ++i) {
        sv[i] = i;
    }
    std::cout << sv.size() << " size sv" << std::endl;

    auto md = stable_index_map<size_t>{};
    while (md.size() < sv.size()) {
        md.grow();
    }
    std::cout << sv.size() << " size md" << std::endl;

    ankerl::nanobench::Bench().minEpochTime(100ms).batch(sv.size()).run("shuffle stable_vector", [&] {
        rng.shuffle(sv);
    });

    ankerl::nanobench::Bench().minEpochTime(100ms).batch(sv.size()).run("shuffle stable_index_map", [&] {
        rng.shuffle(md);
    });

    auto c = std::deque<size_t>();
    for (size_t i = 0; i < sv.size(); ++i) {
        c.push_back(i);
    }
    ankerl::nanobench::Bench().minEpochTime(100ms).batch(sv.size()).run("shuffle std::deque", [&] {
        rng.shuffle(c);
    });

    auto v = std::vector<size_t>();
    for (size_t i = 0; i < sv.size(); ++i) {
        v.push_back(i);
    }
    ankerl::nanobench::Bench().minEpochTime(100ms).batch(sv.size()).run("shuffle std::vector", [&] {
        rng.shuffle(v);
    });
}
