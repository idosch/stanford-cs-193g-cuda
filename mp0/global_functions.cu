#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int *array) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    array[index] = 7;
}

int main(void) {
    int num_elements = 256;
    int num_bytes = num_elements * sizeof(int);

    // pointers to host & device arrays
    int *device_array;
    int *host_array;

    // malloc a host array
    host_array = (int*)malloc(num_bytes);

    //cudaMalloc a device array
    cudaMalloc((void**)&device_array, num_bytes);

    int block_size = 128;
    int grid_size = num_elements / block_size;

    kernel<<<grid_size,block_size>>>(device_array);

    // download a inspect the result on the host
    cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

    // print out the result element by element
    for (int i = 0; i < num_elements; i++) {
        printf("%d ", host_array[i]);
    }

    // deallocate memory
    free(host_array);
    cudaFree(device_array);

    return 0;
}
