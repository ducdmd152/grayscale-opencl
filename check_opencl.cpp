// Main.c
#include <CL/cl.h>
#include <stdio.h>

int main()
{
    cl_uint platformCount;
    cl_platform_id platforms[10];

    cl_int err = clGetPlatformIDs(10, platforms, &platformCount);
    if (err != CL_SUCCESS)
    {
        printf("Failed to get OpenCL platforms.\n");
        return -1;
    }

    printf("Found %u OpenCL platform(s).\n", platformCount);
    return 0;
}
