#ifndef _MGARD_SAMPLE_HPP
#define _MGARD_SAMPLE_HPP

#include <vector>
#include <cstdint>

namespace MGARD{

using namespace std;

template<class T>
void sample_blocks(T const * data, const std::vector<size_t>& dims, std::vector<std::vector<T>>& sampled_blocks, size_t stride=15, size_t block_size=5){
    size_t num_dims = dims.size();
    assert(stride >= 2*block_size);
    switch (num_dims) {
        case 1: {
            if (dims[0] < block_size){
                std::cerr << "Sampling error: block_size greater than data shape" << std::endl;
                exit(-1);
            }
            size_t block_1d_size = block_size;
            std::vector<size_t> starts;
            for(size_t i = block_size; i < dims[0] - block_size; i+=stride){
                starts.push_back(i);
            }
            sampled_blocks.resize(starts.size(), std::vector<T>(block_1d_size));
            for(size_t b = 0; b < starts.size(); b++){
                size_t cur_start = starts[b];
                for(size_t i = 0; i < block_size; i++){
                    size_t idx = cur_start + i;
                    sampled_blocks[b][i] = data[idx];
                }
            }
            break;
        }
        case 2: {
            if (dims[0] < block_size || dims[1] < block_size){
                std::cerr << "Sampling error: block_size greater than data shape" << std::endl;
                exit(-1);
            }
            size_t stride_n1 = dims[1];
            size_t block_2d_size = block_size * block_size;
            std::vector<size_t> starts;
            for(size_t i = block_size; i < dims[0] - block_size; i+=stride){
                for(size_t j = block_size; j < dims[1] - block_size; j+= stride){
                    starts.push_back(i * stride_n1 + j);
                }
            }
            sampled_blocks.resize(starts.size(), std::vector<T>(block_2d_size));
            for(size_t b = 0 ; b < starts.size(); b++){
                size_t cur_start = starts[b];
                size_t count = 0;
                for(size_t i = 0; i < block_size; i++){
                    for(size_t j = 0; j < block_size; j++){
                        size_t idx = cur_start + i * stride_n1 + j;
                        sampled_blocks[b][count++] = data[idx];
                    }
                }
            }
            break;
        }
        case 3: {
            if (dims[0] < block_size || dims[1] < block_size || dims[2] < block_size){
                std::cerr << "Sampling error: block_size greater than data shape" << std::endl;
                exit(-1);
            }
            size_t stride_n1 = dims[1] * dims[2];
            size_t stride_n2 = dims[2];
            size_t block_3d_size = block_size * block_size * block_size;
            std::vector<size_t> starts;
            for(size_t i = block_size; i < dims[0] - block_size; i+=stride){
                for(size_t j = block_size; j < dims[1] - block_size; j+=stride){
                    for(size_t k = block_size; k < dims[2] - block_size; k+=stride){
                        starts.push_back(i * stride_n1 + j * stride_n2 + k);
                    }
                }
            }
            sampled_blocks.resize(starts.size(), std::vector<T>(block_3d_size));
            for(size_t b = 0; b < starts.size(); b++){
                size_t cur_start = starts[b];
                size_t count = 0;
                // if(b < 2) std::cout << "\nblock " << b << ":\n";
                for(size_t i = 0; i < block_size; i++){
                    for(size_t j = 0; j < block_size; j++){
                        for(size_t k = 0; k < block_size; k++){
                            size_t idx = cur_start + i * stride_n1 + j * stride_n2 + k;
                            sampled_blocks[b][count++] = data[idx];
                            // if (b < 2) std::cout << "index within = \t" << count - 1 << ", real index = " << int(idx) << ", value = " << data[idx] << "\n";
                        }
                    }
                }
                // if(b < 2) std::cout << "\n";
            }
            break;
        }
        default:
            break;
    }
}

}
#endif