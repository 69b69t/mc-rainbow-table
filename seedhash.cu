#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "sha256.cuh"

#define CHAINLENGTH (1024ULL*1024ULL)
//#define CHAINLENGTH (20000)

#define BLOCKSIZE (1024ULL*1024ULL)

//#define CHAINCOUNT 1863952114ULL
//#define CHAINCOUNT (1024ULL*1024ULL)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




__device__ __attribute__ ((always_inline, hot)) static inline int next(uint64_t *seed, const int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int) ((int64_t)*seed >> (48 - bits));
}

__device__ __attribute__ ((const, always_inline, hot)) static inline uint64_t nextLong(uint64_t tempSeed)
{
    uint64_t seed = tempSeed;
    return ((uint64_t) next(&seed, 32) << 32) + next(&seed, 32);
}

__device__ __attribute__ ((const, always_inline, hot)) static inline uint64_t seedHash(uint64_t seed)
{
    SHA256_CTX sha;
    sha256_init(&sha);

    //PLATFORM DEPENDANT
    sha256_update(&sha, (uint8_t*)(void*)&seed);
    sha256_final(&sha, (uint8_t*)(void*)&seed);

    //printf("%lx --> %lx\n", originalSeed, seed);
    return seed;
}

__global__ void generateHashChains(uint64_t *tempStart, uint64_t *tempEnd)
{
    //take an array of starting seeds (in tempStart) and perform CHAINLENGTH
    //calculations hash-->nextLong operations on it
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    tempEnd[threadId] = tempStart[threadId];
    for(uint64_t i = 0ULL; i < CHAINLENGTH; i++)
    {
        //function is seed --> hash --> bottom 48 --> newNaturalSeed
        tempEnd[threadId] = nextLong(seedHash(tempEnd[threadId]));
        //tempEnd[threadId] = seedHash(tempEnd[threadId]);
    }
}

int main(int argc, char ** argv)
{
    //input handling
    if(argc != 6)
    {
        printf("Usage: %s <fileName> <chainCount> <startBlock> <endBlock> <gpu>\n\n", argv[0]);
        printf("<fileName>: name of the file you wanna write to\n");
        printf("<chainCount>: the amount of chains to compute\n");
        printf("<startBlock>: the first block of 1024x1024 chains\n");
        printf("<endBlock>: the last block of 1024x1024 chains\n");
        printf("<gpu>: the id of the gpu to run the code on\n");
        return 0;
    }

    char* databaseFile = argv[1];
    uint64_t chainCount = atoi(argv[2]);
    uint32_t startBlock = atoi(argv[3]);
    uint32_t endBlock = atoi(argv[4]);
    uint32_t gpuId = atoi(argv[5]);

    //set device to device id in command
    gpuErrchk(cudaSetDevice(gpuId));

    //---------PRE BRR STUFF----------
    //how big is the data in bytes
    const size_t dataSizeBytes = BLOCKSIZE * sizeof(uint64_t);

    // Allocate host memory for the data
    uint64_t* hostStart = (uint64_t*)malloc(dataSizeBytes);
    uint64_t* hostEnd = (uint64_t*)malloc(dataSizeBytes);

    //Allocate device memory for the data
    uint64_t *seedStart;
    uint64_t *seedEnd;
    gpuErrchk(cudaMalloc((void**)&seedStart, dataSizeBytes));
    gpuErrchk(cudaMalloc((void**)&seedEnd, dataSizeBytes));

    FILE* rainbowTable = fopen(databaseFile, "w");

    printf("need to compute %lld chain blocks\n", chainCount/BLOCKSIZE);

    //loop to generate hash chains in blocks of BLOCKSIZE
    for(uint32_t batchOffset = startBlock; batchOffset <= endBlock; batchOffset++)
    {
        // Initialize hostStart with some values (optional)
        printf("generating input data... %d/%d (%d/%lld)\n", batchOffset-startBlock, endBlock-startBlock, batchOffset, chainCount/BLOCKSIZE);
        for (size_t i = 0; i < BLOCKSIZE; i++) {
            hostStart[i] = i+(BLOCKSIZE * batchOffset);
        }

        // Copy seedStart from host to device
        printf("copying data to device... %d/%d (%d/%lld)\n", batchOffset-startBlock, endBlock-startBlock, batchOffset, chainCount/BLOCKSIZE);
        gpuErrchk(cudaMemcpy(seedStart, hostStart, dataSizeBytes, cudaMemcpyHostToDevice));

        //actually brr the numbers        
        printf("brring numbers... %d/%d (%d/%lld)\n", batchOffset-startBlock, endBlock-startBlock, batchOffset, chainCount/BLOCKSIZE);
        generateHashChains<<<1024, 1024>>>(seedStart, seedEnd);
        
        gpuErrchk( cudaPeekAtLastError() );
        
        //pause execution until gpu finishes executing
        gpuErrchk(cudaDeviceSynchronize());

        // Copy the result back from device to host
        printf("copying data back to host... %d/%d (%d/%lld)\n", batchOffset-startBlock, endBlock-startBlock, batchOffset, chainCount/BLOCKSIZE);
        gpuErrchk(cudaMemcpy(hostEnd, seedEnd, dataSizeBytes, cudaMemcpyDeviceToHost));

        //write computed chains to file
        printf("adding computed data to database... %d/%d (%d/%lld)\n", batchOffset-startBlock, endBlock-startBlock, batchOffset, chainCount/BLOCKSIZE);
        for (uint32_t i = 0; i < BLOCKSIZE; i++) {
            //printf("%lx  --> %lx\n", hostStart[i], hostEnd[i]);
            fwrite(&hostStart[i], sizeof(uint64_t), 1, rainbowTable);
            fwrite(&hostEnd[i], sizeof(uint64_t), 1, rainbowTable);
        }
    }

    //close rainbow table file
    fclose(rainbowTable);

    // Free device memory
    gpuErrchk(cudaFree(seedStart));
    gpuErrchk(cudaFree(seedEnd));

    // Free host memory
    free(hostStart);

    return 0;
}
