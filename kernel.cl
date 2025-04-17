__kernel void rgb_to_grayscale(__global uchar* input, __global uchar* output, int width, int height) {
    int id = get_global_id(0);
    if (id >= width * height) return;

    int index = id * 3;
    uchar r = input[index];
    uchar g = input[index + 1];
    uchar b = input[index + 2];

    uchar gray = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);
    output[id] = gray;
}
