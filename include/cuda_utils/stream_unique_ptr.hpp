// Copyright 2025 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#ifndef CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_
#define CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_

#include <cuda_runtime_api.h>

#include <memory>

namespace cuda_utils
{
struct StreamDeleter
{
    void operator()(cudaStream_t* stream)
    {
        if (stream)
        {
            cudaStreamDestroy(*stream);
            delete stream;
        }
    }
};

using StreamUniquePtr = std::unique_ptr<cudaStream_t, StreamDeleter>;

inline StreamUniquePtr makeCudaStream(const uint32_t flags = cudaStreamDefault)
{
    StreamUniquePtr stream(new cudaStream_t, StreamDeleter());
    if (cudaStreamCreateWithFlags(stream.get(), flags) != cudaSuccess)
    {
        stream.reset(nullptr);
    }
    return stream;
}
}  // namespace cuda_utils

#endif  // CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_
