#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void checkError(cl_int err, const char *operation)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error during operation '" << operation << "', error code: " << err << std::endl;
        exit(1);
    }
}

std::string loadKernel(const char *filename)
{
    std::ifstream file(filename, std::ios::in);
    if (!file.is_open())
    {
        std::cerr << "Failed to load kernel.\n";
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
}

int main()
{
    // Load image
    int width, height, channels;
    unsigned char *input_image = stbi_load("input.jpg", &width, &height, &channels, 3); // Force 3 channels
    if (!input_image)
    {
        std::cerr << "Failed to load image.\n";
        return 1;
    }

    size_t pixel_count = width * height;
    size_t input_size = pixel_count * 3;
    size_t output_size = pixel_count;

    std::vector<unsigned char> output_image(output_size);

    // Load kernel
    std::string kernel_code = loadKernel("kernel.cl");
    const char *kernel_source = kernel_code.c_str();

    // OpenCL Setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    err = clGetPlatformIDs(1, &platform, nullptr);
    // clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) => NVIDIA, AMD, Intel?
    checkError(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    // GPU Device ID
    checkError(err, "clGetDeviceIDs");

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    // Context ~ simply common working space of CPU & GPU contains buffer, program, kernel, cmd queue,...
    checkError(err, "clCreateContext");

    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    // Create buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size, input_image, &err);
    checkError(err, "clCreateBuffer(input)");

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, nullptr, &err);
    checkError(err, "clCreateBuffer(output)");

    // Build program
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size); // get log_size
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr); // get log_data
        std::cerr << "Build log:\n"
                  << build_log.data() << std::endl;
        checkError(err, "clBuildProgram");
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "rgb_to_grayscale", &err);
    checkError(err, "clCreateKernel");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    checkError(err, "clSetKernelArg");

    // Execute kernel
    size_t global_work_size = pixel_count;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    checkError(err, "clEnqueueNDRangeKernel");

    // Read result
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, output_size, output_image.data(), 0, nullptr, nullptr);
    checkError(err, "clEnqueueReadBuffer");

    // Save grayscale image
    stbi_write_jpg("output.jpg", width, height, 1, output_image.data(), 100);
    std::cout << "Grayscale image saved as output.jpg\n";

    // Cleanup
    stbi_image_free(input_image);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
