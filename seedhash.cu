#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <sqlite3.h>

#include "sha256.cuh"

#define DATABASE_NAME "rainbowTable.db"
#define MASK48 0xffffffffffff

sqlite3 *db;

__device__ static inline int next(uint64_t *seed, const int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int) ((int64_t)*seed >> (48 - bits));
}

__device__ static inline uint64_t nextLong(uint64_t tempSeed)
{
    uint64_t seed = tempSeed;
    return ((uint64_t) next(&seed, 32) << 32) + next(&seed, 32);
}

__device__ __attribute__ ((const, always_inline, hot)) inline uint64_t seedHash(uint64_t seed)
{
    uint8_t byteArray[32] = {0};
    byteArray[0] = seed & 0xff;
    byteArray[1] = (seed >> 8) & 0xff;
    byteArray[2] = (seed >> 16) & 0xff;
    byteArray[3] = (seed >> 24) & 0xff;
    byteArray[4] = (seed >> 32) & 0xff;
    byteArray[5] = (seed >> 40) & 0xff;
    byteArray[6] = (seed >> 48) & 0xff;
    byteArray[7] = seed >> 56;

    SHA256_CTX sha;
    sha256_init(&sha);
    sha256_update(&sha, byteArray, 8);
    sha256_final(&sha, byteArray);

    seed = 0;
    for(int i = 0; i < 8; i++)
    {
        seed <<= 8;
        seed += byteArray[7 - i];
    }

    return seed;
}

__global__ void getEndOfChain(uint64_t *tempStart, uint64_t *tempEnd)
{
    const uint64_t chainLength = 7794473ULL;
    //const uint64_t chainCount = 649885037ULL;
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    tempEnd[threadId] = tempStart[threadId];
    for(uint64_t i = 0ULL; i < chainLength; i++)
    {
        //function is seed --> hash --> bottom 48 --> newNaturalSeed
        tempEnd[threadId] = nextLong(seedHash(tempEnd[threadId]));
    }
}

static inline int hostNext(uint64_t *seed, const int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int) ((int64_t)*seed >> (48 - bits));
}

static inline uint64_t hostNextLong(uint64_t tempSeed)
{
    uint64_t seed = tempSeed;
    return ((uint64_t) hostNext(&seed, 32) << 32) + hostNext(&seed, 32);
}

void createDatabase()
{
    int rc;
    rc = sqlite3_exec(db, "CREATE TABLE rainbowTable(startSeed INTEGER, endSeed INTEGER);", NULL, 0, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        exit(1);
    }

    rc = sqlite3_exec(db, "CREATE INDEX hashIndex ON rainbowTable(endSeed);", NULL, 0, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        exit(1);
    }

    rc = sqlite3_exec(db, "PRAGMA synchronous = OFF;", NULL, 0, NULL);
    rc = sqlite3_exec(db, "PRAGMA journal_mode = WAL;", NULL, 0, NULL);
}

void addToDatabase(uint64_t startSeed, uint64_t endSeed)
{
    char sql[128] = {0};
    sprintf(sql, "INSERT INTO rainbowTable(startSeed, endSeed) VALUES(%ld, %ld);", startSeed, endSeed);

    int rc;
    rc = sqlite3_exec(db, sql, NULL, 0, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        exit(1);
    }
}

int main() {
    //---------PRE BRR STUFF----------
    //how big is the data in bytes
    const size_t dataSizeBytes = 1048576 * sizeof(uint64_t);

    // Allocate host memory for the data
    uint64_t* hostStart = (uint64_t*)malloc(dataSizeBytes);
    uint64_t* hostEnd = (uint64_t*)malloc(dataSizeBytes);

    // Allocate device memory for the data
    uint64_t *seedStart;
    uint64_t *seedEnd;
    cudaMalloc((void**)&seedStart, dataSizeBytes);
    cudaMalloc((void**)&seedEnd, dataSizeBytes);

    //sqlite3 stuff
    sqlite3_open(DATABASE_NAME, &db);
    createDatabase();


    //----------BRR LOOP---------
    for(uint32_t batchOffset = 0; batchOffset < 620; batchOffset++)
    {
        // Initialize hostStart with some values (optional)
        printf("generating input data... %d/620\n", batchOffset);
        for (size_t i = 0; i < 1048576; i++) {
            hostStart[i] = hostNextLong(i+(1048576ULL * batchOffset));
        }

        // Copy seedStart from host to device
        printf("copying data to device... %d/620\n", batchOffset);
        cudaMemcpy(seedStart, hostStart, dataSizeBytes, cudaMemcpyHostToDevice);
        
        printf("brring numbers... %d/620\n", batchOffset);
        getEndOfChain<<<1024, 1024>>>(seedStart, seedEnd);

        //get last error and stuff
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("Error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
        
        cudaDeviceSynchronize();

        // Copy the result back from device to host
        printf("copying data back to host... %d/620\n", batchOffset);
        cudaMemcpy(hostEnd, seedEnd, dataSizeBytes, cudaMemcpyDeviceToHost);

        printf("adding computed data to database... %d/620\n", batchOffset);
        for (size_t i = 0; i < 1048576; i++) {
            //printf("Element %zu: %lu\n", i, hostStart[i]);
            addToDatabase(hostStart[i], hostEnd[i]);
        }
    }
    //------------END BRR LOOP-------------


    printf("analyzing database...\n");
    sqlite3_exec(db, "ANALYZE;", NULL, 0, NULL);

    printf("indexing hashes...\n");
    sqlite3_exec(db, "CREATE INDEX hashIndex ON rainbowTable(endSeed);", NULL, 0, NULL);
    sqlite3_close(db);

    // Free device memory
    cudaFree(seedStart);
    cudaFree(seedEnd);

    // Free host memory
    free(hostStart);

    return 0;
}