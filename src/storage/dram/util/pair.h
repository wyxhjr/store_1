#ifndef UTIL_PAIR_H_
#define UTIL_PAIR_H_

#include <cstdint>
#include <cstdlib>

typedef uint64_t Key_t;
typedef uint64_t Value_t;

const Key_t SENTINEL = UINT64_MAX - 1; // 11111...110
const Key_t INVALID = UINT64_MAX;      // 11111...111

const Value_t NONE = 0;

struct Pair {
  Key_t key;
  Value_t value;

  Pair(void) : key{INVALID} {}

  Pair(Key_t _key, Value_t _value) : key{_key}, value{_value} {}

  Pair &operator=(const Pair &other) {
    key = other.key;
    value = other.value;
    return *this;
  }

  void *operator new(size_t size) {
    void *ret;
    posix_memalign(&ret, 64, size);
    return ret;
  }

  void *operator new[](size_t size) {
    void *ret;
    posix_memalign(&ret, 64, size);
    return ret;
  }
};

#endif // UTIL_PAIR_H_