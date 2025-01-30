#include "tensorstore/driver/hdf5/error_handling.h"

#include "absl/status/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace hdf5_driver {

thread_local std::string HDF5ErrorHandler::last_error_;

HDF5ErrorHandler::HDF5ErrorHandler() {
    // Save old error handler
    H5Eget_auto2(H5E_DEFAULT, &old_func_, &old_client_data_);
    
    // Set new error handler
    H5Eset_auto2(H5E_DEFAULT, &HDF5ErrorHandler::HandleError, nullptr);
}

HDF5ErrorHandler::~HDF5ErrorHandler() {
    // Restore old error handler
    H5Eset_auto2(H5E_DEFAULT, old_func_, old_client_data_);
}

herr_t HDF5ErrorHandler::HandleError(hid_t stack_id, void* client_data) {
    // Get error details
    H5E_type_t error_type;
    ssize_t msg_len = H5Eget_msg(stack_id, &error_type, nullptr, 0);
    if (msg_len <= 0) {
        last_error_ = "Unknown HDF5 error";
        return 0;
    }
    
    std::string error_msg(msg_len + 1, '\0');
    H5Eget_msg(stack_id, &error_type, error_msg.data(), msg_len + 1);
    
    // Store error message
    last_error_ = error_msg;
    
    // Continue error stack traversal
    return 0;
}

Status HDF5ErrorHandler::ToStatus(const std::string& context) {
    if (last_error_.empty()) {
        return absl::OkStatus();
    }
    
    // Map HDF5 errors to appropriate TensorStore status codes
    // TODO: Add more specific error mappings based on error type
    return absl::InternalError(tensorstore::StrCat(context, ": ", last_error_));
}

}  // namespace hdf5_driver
}  // namespace tensorstore
