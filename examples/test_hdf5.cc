#include "tensorstore/tensorstore.h"
#include <hdf5.h>
#include <iostream>

void create_hdf5_file(const std::string& filename) {
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error creating file" << std::endl;
        return;
    }
    H5Fclose(file_id);
}

int main() {
    create_hdf5_file("test.h5");
    std::cout << "HDF5 file created successfully" << std::endl;
    return 0;
}
