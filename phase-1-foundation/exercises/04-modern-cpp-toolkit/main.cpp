// ============================================================
// 04 - Modern C++ toolkit: RAII, moves, templates, lambdas [STARTER]
// ============================================================

#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "reference.h"
#include "report.h"
#include "timer.h"
#include "verify.h"

// ============================================================
// Matrix<T> - an owning, movable, templated 2D array
// ============================================================
// This is the host-side rehearsal for Phase 2 exercise 12, where
// the same structure wraps cudaMalloc / cudaFree instead of new[].
template <typename T>
class Matrix {
public:
    Matrix() = default;

    // TODO 1: allocate rows * cols elements.
    // Use std::make_unique<T[]>(n) - it zero-initialises and it is
    // exception safe. Call AllocationTracker::onAllocate() so the
    // tests can see the allocation happen.
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        (void)rows;
        (void)cols;
        // TODO
    }

    // TODO 2: destructor.
    // unique_ptr frees the storage automatically, so the only work
    // left is telling the tracker - but only if we still own data.
    ~Matrix() {
        // TODO
    }

    // TODO 3: copy constructor - a DEEP copy.
    // Allocate fresh storage and copy every element. This is the
    // expensive operation we want to be able to avoid.
    Matrix(const Matrix& other) {
        (void)other;
        // TODO
    }

    // TODO 4: move constructor.
    // Steal `other`'s pointer instead of copying the data, then
    // leave `other` empty so its destructor does nothing. Mark it
    // noexcept: std::vector only uses moves when growing if the
    // move constructor promises not to throw.
    Matrix(Matrix&& other) noexcept {
        (void)other;
        // TODO
    }

    // TODO 5: copy assignment and move assignment.
    Matrix& operator=(const Matrix& other) {
        (void)other;
        // TODO
        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        (void)other;
        // TODO
        return *this;
    }

    // TODO 6: element access, row-major -> data_[r * cols_ + c].
    T& operator()(size_t r, size_t c) {
        (void)r;
        (void)c;
        return data_[0];  // TODO
    }
    const T& operator()(size_t r, size_t c) const {
        (void)r;
        (void)c;
        return data_[0];  // TODO
    }

    // TODO 7: apply a callable to every element.
    // Templated on the callable so the compiler can inline the
    // lambda - no std::function, no indirect call, no cost.
    template <typename Fn>
    void apply(Fn fn) {
        (void)fn;
        // TODO
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

// A factory returning by value. Without a move constructor this
// would deep-copy the result; with one it costs three pointer
// assignments.
template <typename T>
Matrix<T> makeRamp(size_t rows, size_t cols) {
    Matrix<T> m(rows, cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            m(r, c) = static_cast<T>((r * cols + c) * 0.5);
    return m;
}

int main() {
    printBanner("Phase 1 / 04 - Modern C++ toolkit (starter)");

    AllocationTracker::reset();
    Matrix<float> probe(4, 4);
    if (AllocationTracker::total == 0 || probe.empty()) {
        printTodoNotice("implement Matrix<T> in main.cpp (TODO 1-7)");
        return verifySummary();
    }

    printSection("Correctness");
    // TODO 8: write the checks yourself, or read solution.cpp to
    // see which properties are worth asserting. At minimum:
    //   - element access round-trips
    //   - a move performs no new allocation
    //   - a moved-from matrix is empty
    //   - nothing leaks (AllocationTracker::live == 0 at the end)

    return verifySummary();
}
