#ifndef _MGARD_DECOMPOSE_INTERLEAVE_NEW_HPP
#define _MGARD_DECOMPOSE_INTERLEAVE_NEW_HPP

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include "reorder.hpp"
#include "utils.hpp"
#include "correction.hpp"

namespace MGARD{

using namespace std;

template <class T>
class Decomposer_Interleaver_new{
public:
	Decomposer_Interleaver_new(bool use_sz_=true){
            use_sz = use_sz_;
        };
	~Decomposer_Interleaver_new(){
	};
    // return decomposed and interleaved data buffers for each level, num_levels = Dims * MGARD_level (target_level)
	// Combining MGARD Decomposer and Interleaver
	std::vector<std::vector<T>> decompose(T * data, const vector<size_t>& dims, size_t target_level, bool hierarchical=false, bool cubic=false, vector<size_t> strides=vector<size_t>()){
		data_begin = data;
		size_t num_elements = 1;
		for(const auto& d:dims){
			num_elements *= d;
		}
        int max_level = log2(*min_element(dims.begin(), dims.end()));
        if(target_level > max_level) target_level = max_level;
		init(dims, target_level);
		if(dims.size() == 1){
			size_t h = 1;
			size_t n = dims[0];
			for(int current_level=target_level; current_level >= 0; current_level--){
				decompose_interleave_level_1D_with_hierarchical_basis(data, n, h, current_level);
				h <<= 1;
			}
		}
		else if(dims.size() == 2){
			size_t h = 1;
			size_t n1 = dims[0];
			size_t n2 = dims[1];
			for(int current_level=target_level; current_level >= 0; current_level--){
				decompose_interleave_2D_with_hierarchical_basis(data, n1, n2, h, current_level);
				h <<= 1;
			}
		}
		else if(dims.size() == 3){
			size_t h = 1;
			size_t n1 = dims[0];
			size_t n2 = dims[1];
			size_t n3 = dims[2];
			for(int current_level=target_level; current_level >= 0; current_level--){
				decompose_interleave_3D_with_hierarchical_basis(data, n1, n2, n3, h, current_level);
				h <<= 1;
			}
		}
        return std::move(level_buffers);
	}
	std::vector<std::vector<uint32_t>> get_level_buffer_dims(){
		return level_buffer_dims;
	}

private:
    bool use_sz = true;
	std::vector<std::vector<T>> level_buffers;
	std::vector<std::vector<uint32_t>> level_buffer_dims;
	std::vector<uint32_t> level_sizes;
	T * data_begin;

	void init(const vector<size_t>& dims, size_t target_level){
		std::vector<uint32_t> dims_uint32(dims.begin(), dims.end());
		auto level_dims = compute_level_dims_new(dims_uint32, target_level);
		level_sizes = compute_level_buffers_size(level_dims, target_level, level_buffer_dims);
		for(int i=0; i<level_sizes.size(); i++){
			level_buffers.push_back(std::vector<T>(level_sizes[i]));
		}
	}

	size_t interleave_1D_level_0(const size_t begin, const size_t end, const size_t stride, T * data, T * buffer){
		size_t count = 0;
		for(size_t i=begin; i<=end; i+=stride){
			buffer[count++] = data[i];
		}
		return count;
	}

	size_t compute_interpolant_difference_1D(const size_t begin, const size_t end, const size_t stride, T * data, T * buffer){
		size_t n = (end - begin) / stride + 1;
		size_t i = 1;
		size_t count = 0;
		for(i=1; i+1<n; i+=2){
			size_t c = begin + i * stride;
			// buffer[c] -= (buffer[c - stride] + buffer[c + stride]) / 2;
			// if((&data[c] - data_begin) == 2) std::cout << (&data[c] - data_begin) << " -= (" << (&data[c - stride] - data_begin) << " + " << (&data[c + stride] - data_begin) << ")/2" << std::endl;
			buffer[count++] = data[c] - (data[c - stride] + data[c + stride]) / 2;
		}
		if(n % 2 == 0){
			size_t c = begin + (n - 1) * stride;
			// buffer[c] -= buffer[c - stride];
			buffer[count++] = data[c] - data[c - stride];
		}
		return count;
	}

	void decompose_interleave_level_1D_with_hierarchical_basis(T * data_pos, size_t n, T h, size_t current_level){
		size_t count_1D = 0;
		size_t buffer_index = current_level;
		count_1D = (current_level) ? compute_interpolant_difference_1D(0, n-1, h, data_pos, level_buffers[buffer_index].data()) : interleave_1D_level_0(0, n-1, h, data_pos, level_buffers[buffer_index].data());
		assert(count_1D == level_sizes[current_level]);
    }

	size_t interleave_2D_level_0(T * data_pos, size_t n1, size_t n2, size_t h){
		size_t count_2D = 0;
		size_t count;
		size_t stride_n2 = h;
		size_t stride_n1 = n2 * h;

		T * cur_data_pos = data_pos;
		T * cur_buffer_pos = level_buffers[0].data();
		size_t n2_begin = 0;
		size_t n2_end = n2 - 1;
		for(size_t i=0; i<n1; i+=h){
			count = interleave_1D_level_0(n2_begin, n2_end, stride_n2, cur_data_pos, cur_buffer_pos);
			cur_data_pos += stride_n1;
			cur_buffer_pos += count;
			count_2D += count;
		}
		return count_2D;
	}

	size_t compute_interpolant_difference_2D(T * data_pos, size_t n1, size_t n2, size_t h, size_t current_level){
		size_t count_2D = 0;
		size_t count;
		size_t stride_n1 = n2 * h;
		size_t stride_n2 = h;

		T * cur_data_pos = data_pos;
		T * cur_buffer_pos = level_buffers[((current_level - 1) * 2) + 2].data();
		size_t n2_begin = 0;
		size_t n2_end = n2 - 1;
		for(size_t i=0; i<n1; i+=h){
			count = compute_interpolant_difference_1D(n2_begin, n2_end, stride_n2, cur_data_pos, cur_buffer_pos);
			cur_data_pos += n2 * h;
			cur_buffer_pos += count;
			count_2D += count;
		}
		// compute vertical difference
		cur_data_pos = data_pos;
		cur_buffer_pos = level_buffers[((current_level - 1) * 2) + 1].data();
		h = h << 1;
		size_t n1_begin = 0;
		size_t n1_end = (n1 - 1) * n2;
		for(size_t i=0; i<n2; i+=h){
			count = compute_interpolant_difference_1D(n1_begin, n1_end, stride_n1, cur_data_pos, cur_buffer_pos);
			cur_data_pos += h;
			cur_buffer_pos += count;
			count_2D += count;
		}
		return count_2D;
	}

	void decompose_interleave_2D_with_hierarchical_basis(T * data_pos, size_t n1, size_t n2, size_t h, size_t current_level){
        size_t count_2D = 0;
		count_2D = (current_level) ? compute_interpolant_difference_2D(data_pos, n1, n2, h, current_level) : interleave_2D_level_0(data_pos, n1, n2, h);
		assert(count_2D == (current_level) ? level_sizes[((current_level - 1) * 2) + 1] + level_sizes[((current_level - 1) * 2) + 2] : level_sizes[0]);
    }

	size_t interleave_3D_level_0(T * data_pos, size_t n1, size_t n2, size_t n3, size_t h){
		size_t count_3D = 0;
		size_t count;
		size_t stride_n3 = h;
		size_t stride_n2 = n3*h;
		size_t stride_n1 = n2 * n3 * h;

		T * cur_data_pos = data_pos;
		T * temp_data_pos = cur_data_pos;
		T * cur_buffer_pos = level_buffers[0].data();
		size_t n3_begin = 0;
		size_t n3_end = n3 - 1;
		for(size_t i=0; i<n1; i+=h){
			temp_data_pos = cur_data_pos;
			for(size_t j=0; j<n2; j+=h){
				count = interleave_1D_level_0(n3_begin, n3_end, h, temp_data_pos, cur_buffer_pos);
				temp_data_pos += n3*h;
				cur_buffer_pos += count;
				count_3D += count;
			}
			cur_data_pos += n2*n3*h;
		}

		return count_3D;
	}

	size_t compute_interpolant_difference_3D(T * data_pos, size_t n1, size_t n2, size_t n3, size_t h, size_t current_level){
		size_t count_3D = 0;
		size_t count;
		size_t stride_n1 = n2 * n3 * h;
		size_t stride_n2 = n3 * h;

		T * cur_data_pos = data_pos;
		T * temp_data_pos = cur_data_pos;
		T * cur_buffer_pos = level_buffers[((current_level - 1) * 3) + 3].data();
		size_t n3_begin = 0;
		size_t n3_end = n3 - 1;
		for(size_t i=0; i<n1; i+=h){
			temp_data_pos = cur_data_pos;
			for(size_t j=0; j<n2; j+=h){
				count = compute_interpolant_difference_1D(n3_begin, n3_end, h, temp_data_pos, cur_buffer_pos);
				temp_data_pos += n3*h;
				cur_buffer_pos += count;
				count_3D += count;
			}
			cur_data_pos += n2*n3*h;
		}

		size_t h2x = h << 1;
		cur_data_pos = data_pos;
		cur_buffer_pos = level_buffers[((current_level - 1) * 3) + 2].data();
		size_t n2_begin = 0;
		size_t n2_end = (n2-1) * n3;
		for(size_t i=0; i<n1; i+=h){
			temp_data_pos = cur_data_pos;
			for(size_t j=0; j<n3; j+=h2x){
				count = compute_interpolant_difference_1D(n2_begin, n2_end, stride_n2, temp_data_pos, cur_buffer_pos);
				temp_data_pos += h2x;
				cur_buffer_pos += count;
				count_3D += count;
			}
			cur_data_pos += n2*n3*h;
		}

		// h = h << 1;
		cur_data_pos = data_pos;
		cur_buffer_pos = level_buffers[((current_level - 1) * 3) + 1].data();
		size_t n1_begin = 0;
		size_t n1_end = (n1-1) * n2 * n3;
		for(size_t i=0; i<n2; i+=h2x){
			temp_data_pos = cur_data_pos;
			for(size_t j=0; j<n3; j+=h2x){
				count = compute_interpolant_difference_1D(n1_begin, n1_end, stride_n1, temp_data_pos, cur_buffer_pos);
				temp_data_pos += h2x;
				cur_buffer_pos += count;
				count_3D += count;
			}
			cur_data_pos += n3 * h2x;
		}
		return count_3D;
	}

	void decompose_interleave_3D_with_hierarchical_basis(T * data_pos, size_t n1, size_t n2, size_t n3, T h, size_t current_level){
		size_t count_3D = 0;
        count_3D = (current_level != 0) ? compute_interpolant_difference_3D(data_pos, n1, n2, n3, h, current_level) : interleave_3D_level_0(data_pos, n1, n2, n3, h);
		assert(count_3D == (current_level) ? level_sizes[((current_level - 1) * 3) + 1] + level_sizes[((current_level - 1) * 3) + 2] + level_sizes[((current_level - 1) * 3) + 3] : level_sizes[0]);
    }
};
}

#endif