#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <iomanip>
#include <cmath>
#include "decompose_interleave_new.hpp"
#include "reposition_recompose_new.hpp"

using namespace std;

std::vector<uint32_t> interp_order;

template <class T>
std::vector<std::vector<T>> test_decompose(vector<T>& data, const vector<size_t>& dims, int target_level){
    struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    MGARD::Decomposer_Interleaver_new<T> decomposer;
    decomposer.interp_order = interp_order;
    std::vector<std::vector<T>> level_buffers = decomposer.decompose(data.data(), dims, target_level, false, true);
    err = clock_gettime(CLOCK_REALTIME, &end);
    cout << "Decomposition time: " << (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000 << "s" << endl;
    return level_buffers;
}

template <class T>
std::vector<T> test_recompose(std::vector<std::vector<T>>& level_buffers, const vector<size_t>& dims, int target_level){
    struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    MGARD::Repositioner_Recomposer_new<T> recomposer;
    recomposer.interp_order = interp_order;
    std::vector<T> recovered_data = recomposer.recompose(level_buffers, dims, target_level, false, true);
    err = clock_gettime(CLOCK_REALTIME, &end);
    cout << "Recomposition time: " << (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000 << "s" << endl;
    return recovered_data;
}

template <class T>
void test(string filename, const vector<size_t>& dims, int target_level){
    size_t num_elements = 0;
    auto data = MGARD::readfile<T>(filename.c_str(), num_elements);
    auto data_ori(data);
    auto level_buffers = test_decompose(data, dims, target_level);
    double score = 0.0;
    for (size_t l = 1; l < level_buffers.size(); l++) {
        const auto& buf = level_buffers[l];
        for (size_t i = 0; i < buf.size(); i++) {
            score += std::log1p(std::abs(buf[i]) + 1.0);
        }
    }
    std::cout << "score = " << score << std::endl;
    auto recovered_data = test_recompose(level_buffers, dims, target_level);
    // struct timespec start, end;
    // int err = 0;
    // err = clock_gettime(CLOCK_REALTIME, &start);
    // MGARD::Decomposer_Interleaver<T> decomposer;
    // std::vector<std::vector<T>> level_buffers = decomposer.decompose(data.data(), dims, target_level, true);
    // err = clock_gettime(CLOCK_REALTIME, &end);
    // cout << "Decomposition time: " << (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000 << "s" << endl;
    // err = clock_gettime(CLOCK_REALTIME, &start);
    // MGARD::Repositioner_Recomposer<T> recomposer;
    // std::vector<T> recovered_data = recomposer.recompose(level_buffers, dims, target_level, true);
    // err = clock_gettime(CLOCK_REALTIME, &end);
    // cout << "Recomposition time: " << (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000 << "s" << endl;
    MGARD::print_statistics(data_ori.data(), recovered_data.data(), num_elements);
}

int main(int argc, char ** argv){
    int argv_id = 1;
    string filename = string(argv[argv_id++]);
    int type = atoi(argv[argv_id++]); // 0 for float, 1 for double
    int target_level = atoi(argv[argv_id++]);
    const int num_dims = atoi(argv[argv_id++]);
    vector<size_t> dims(num_dims);
    for(int i=0; i<dims.size(); i++){
       dims[i] = atoi(argv[argv_id++]);
       cout << dims[i] << " ";
    }
    cout << endl;
    interp_order.resize(num_dims);
    for(int i=0; i<dims.size(); i++){
        interp_order[i] = atoi(argv[argv_id++]);
    }

    switch(type){
        case 0:
            {
                test<float>(filename, dims, target_level);
                break;
            }
        case 1:
            {
                test<double>(filename, dims, target_level);
                break;
            }
        default:
            cerr << "Only 0 (float) and 1 (double) are implemented in this test\n";
            exit(0);
    }
    return 0;
}