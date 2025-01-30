#ifndef TENSORSTORE_DRIVER_HDF5_ERROR_HANDLING_H_
#define TENSORSTORE_DRIVER_HDF5_ERROR_HANDLING_H_

#include <string>
#include "absl/strings/str_cat.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace hdf5_driver {

/// RAII class for managing HDF5 error handling
class HDF5ErrorHandler {
public:
    HDF5ErrorHandler();
    ~HDF5ErrorHandler();

    /// Callback function for HDF5 error handling
    static herr_t HandleError(hid_t stack_id, void* client_data);

    /// Gets the last error message
    static std::string GetLastError() { return last_error_; }

    /// Converts HDF5 error to TensorStore status
    static Status ToStatus(const std::string& context);

private:
    static thread_local std::string last_error_;
    herr_t (*old_func_)(hid_t, void*);
    void* old_client_data_;
};

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_ERROR_HANDLING_H_
