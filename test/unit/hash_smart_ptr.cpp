#include <ankerl/unordered_dense.h>

#include <doctest.h>

#include <memory>

template <typename Ptr>
void check(Ptr const& ptr) {
    REQUIRE(ankerl::unordered_dense::hash<Ptr>{}(ptr) ==
            ankerl::unordered_dense::hash<decltype(std::declval<Ptr>().get())>{}(ptr.get()));
}

TEST_CASE("hash_smart_ptr") {
    check(std::unique_ptr<uint64_t>{});
    check(std::shared_ptr<uint64_t>{});
    check(std::make_shared<uint64_t>(123U));
    check(std::make_unique<uint64_t>(123U));
    check(std::make_unique<uint64_t>(uint64_t{123U}));
}
