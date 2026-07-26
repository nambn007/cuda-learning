// ============================================================
// 04 - Modern C++ toolkit: RAII, moves, templates, lambdas [SOLUTION]
// ============================================================

#include <algorithm>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

template <typename T>
class Matrix {
public:
    Matrix() = default;

    Matrix(size_t rows, size_t cols)
        : data_(rows * cols ? std::make_unique<T[]>(rows * cols) : nullptr),
          rows_(rows),
          cols_(cols) {
        if (data_) AllocationTracker::onAllocate();
    }

    // The destructor does not call delete[]: unique_ptr does that.
    // This is RAII - ownership is encoded in the type, so there is
    // no code path, not even an exception, that can leak.
    ~Matrix() {
        if (data_) AllocationTracker::onFree();
    }

    // Copy: allocate new storage and duplicate the contents.
    Matrix(const Matrix& other)
        : data_(other.data_ ? std::make_unique<T[]>(other.size()) : nullptr),
          rows_(other.rows_),
          cols_(other.cols_) {
        if (data_) {
            AllocationTracker::onAllocate();
            std::copy(other.data_.get(), other.data_.get() + other.size(), data_.get());
        }
    }

    // Move: take the pointer, leave the source empty.
    //
    // noexcept is not decoration. std::vector will only move its
    // elements when it reallocates if the move constructor is
    // noexcept; otherwise it copies, to keep its strong exception
    // guarantee. A missing noexcept silently costs you deep copies.
    Matrix(Matrix&& other) noexcept
        : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_) {
        // A moved-from object must stay valid and destructible.
        other.rows_ = 0;
        other.cols_ = 0;
    }

    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;  // self-assignment
        Matrix tmp(other);                 // copy-and-swap: strong guarantee
        swap(tmp);
        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this == &other) return *this;
        if (data_) AllocationTracker::onFree();
        data_ = std::move(other.data_);
        rows_ = other.rows_;
        cols_ = other.cols_;
        other.rows_ = 0;
        other.cols_ = 0;
        return *this;
    }

    void swap(Matrix& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
    }

    T& operator()(size_t r, size_t c) { return data_[r * cols_ + c]; }
    const T& operator()(size_t r, size_t c) const { return data_[r * cols_ + c]; }

    // Templated on the callable, not std::function. The lambda's
    // body is inlined into this loop, so `apply` compiles down to
    // exactly the same machine code as a hand-written loop.
    // Thrust and CUB use the same technique for device functors.
    template <typename Fn>
    void apply(Fn fn) {
        const size_t n = size();
        for (size_t i = 0; i < n; ++i) data_[i] = fn(data_[i]);
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    bool empty() const { return data_ == nullptr; }
    T* data() { return data_.get(); }
    const T* data() const { return data_.get(); }

private:
    std::unique_ptr<T[]> data_;
    size_t rows_ = 0;
    size_t cols_ = 0;
};

template <typename T>
Matrix<T> makeRamp(size_t rows, size_t cols) {
    Matrix<T> m(rows, cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            m(r, c) = static_cast<T>((r * cols + c) * 0.5);
    return m;
}

template <typename T>
double sumOf(const Matrix<T>& m) {
    double s = 0.0;
    for (size_t r = 0; r < m.rows(); ++r)
        for (size_t c = 0; c < m.cols(); ++c) s += static_cast<double>(m(r, c));
    return s;
}

int main() {
    printBanner("Phase 1 / 04 - Modern C++ toolkit");

    const size_t R = 256, C = 256;
    AllocationTracker::reset();

    printSection("RAII and element access");
    {
        Matrix<float> m = makeRamp<float>(R, C);
        reportCheck("constructor allocated once", AllocationTracker::total == 1);
        reportCheck("element access round-trips",
                    m(3, 7) == static_cast<float>((3 * C + 7) * 0.5));
        reportCheck("sum matches expectation",
                    std::abs(sumOf(m) - expectedSum(R, C)) < 1e-3);
    }
    // m went out of scope; the destructor ran without us writing
    // a single delete.
    reportCheck("nothing leaked after scope exit", AllocationTracker::live == 0);

    printSection("Copy versus move");
    AllocationTracker::reset();
    {
        Matrix<float> original = makeRamp<float>(R, C);
        long long afterOriginal = AllocationTracker::total;

        Matrix<float> copied = original;  // deep copy
        reportCheck("copy allocates new storage",
                    AllocationTracker::total == afterOriginal + 1);
        reportCheck("copy is independent",
                    copied.data() != original.data() && copied(1, 1) == original(1, 1));

        long long beforeMove = AllocationTracker::total;
        Matrix<float> moved = std::move(copied);  // steal
        reportCheck("move allocates nothing", AllocationTracker::total == beforeMove);
        reportCheck("moved-from matrix is empty", copied.empty() && copied.size() == 0);
        reportCheck("moved-to matrix holds the data", moved(1, 1) == original(1, 1));
    }
    reportCheck("no leaks after copy/move scope", AllocationTracker::live == 0);

    printSection("Templates and lambdas");
    {
        // The same class, instantiated for a different type.
        Matrix<double> d(4, 4);
        d.apply([](double) { return 2.5; });
        reportCheck("Matrix<double> works", d(2, 2) == 2.5);

        Matrix<int> i(4, 4);
        i.apply([](int) { return 7; });
        // A capturing lambda is just as inlinable as a stateless one.
        int bias = 3;
        i.apply([bias](int v) { return v + bias; });
        reportCheck("capturing lambda applied", i(0, 0) == 10);
    }

    printSection("Zero-cost abstraction");
    // If Matrix<T> really costs nothing, a loop through operator()
    // should run at the same speed as a loop over a raw pointer.
    Matrix<float> big(2048, 2048);
    std::vector<float> raw(2048ull * 2048ull, 1.0f);

    double msClass = timeCpuMs(10, [&] {
        big.apply([](float v) { return v * 1.000001f + 1.0f; });
    });
    double msRaw = timeCpuMs(10, [&] {
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = raw[i] * 1.000001f + 1.0f;
    });

    ResultTable table;
    table.add("raw std::vector loop", msRaw, raw.size() * sizeof(float) * 2);
    table.add("Matrix<T>::apply(lambda)", msClass, big.size() * sizeof(float) * 2);
    table.print("Abstraction overhead");
    printf("\n  The two rows should be within noise of each other. Templates and\n");
    printf("  lambdas are resolved at compile time - there is no virtual call\n");
    printf("  and nothing left to pay for at run time.\n");

    // Exactly one matrix (`big`) is still in scope, so exactly one
    // allocation should be outstanding - no strays from the loops
    // above, no double frees.
    reportCheck("only the live matrix is still allocated", AllocationTracker::live == 1);
    return verifySummary();
}
