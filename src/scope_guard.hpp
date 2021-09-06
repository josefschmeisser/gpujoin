#pragma once

#include <functional>
#include <utility>

template<typename T, typename Closer>
struct scope_guard {
private:
    T guardee_;
    Closer closer_;

public:
    template<typename CloserResult>
    scope_guard(const T guardee, std::function<CloserResult(T)>&& closer) noexcept : guardee_(guardee), closer_(closer) {}

    ~scope_guard() {
        Closer(guardee);
    }

    T get() const noexcept {
        return guardee_;
    }

    template<typename CloserResult>
    T release_impl(std::function<CloserResult(T)>&) noexcept {
        // assign a no-op function
        closer_ = [](T) { return CloserResult(); };
        return guardee_;
    }

    T release() noexcept {
        return release_impl(closer_);
    }
};

template<typename T, typename Closer>
auto make_scope_guard(const T guardee, const T invalid_value, Closer&& closer) {
    using fun_type = std::function<decltype(closer(invalid_value))(T)>;
    if (guardee != invalid_value) {
        return scope_guard<T, fun_type>(guardee, fun_type(closer));
    } else {
        // return a no-op guard
        using return_type = decltype(closer(invalid_value));
        return scope_guard<T, fun_type>(guardee, fun_type([](T) -> return_type { return return_type(); }));
    }
}

template<typename T, typename Closer>
auto make_scope_guard(const T guardee, Closer&& closer) {
    using fun_type = std::function<decltype(closer(std::declval<T>()))(T)>;
    return scope_guard<T, fun_type>(guardee, fun_type(closer));
}
