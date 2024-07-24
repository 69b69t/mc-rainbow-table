#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include "sha256.h"

//bytes divided by sizeof(uint64_t) * 2
#define CHAINLENGTH (500000ULL)
#define THREADCOUNT 16

typedef struct{
    uint64_t index;
    uint64_t input;
    uint64_t hash;
} DataCell;

//i guess this is another way to write the same thing?
//fix the aboove one
struct threadedParameters{
    uint64_t* chainCanidates;
    uint64_t chainCanidatesCount;
    uint64_t targetHash;
    uint32_t reducerCount;
    uint32_t thread;
};

static inline int next(uint64_t *seed, const int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int) ((int64_t)*seed >> (48 - bits));
}

static inline uint64_t nextLong(uint64_t tempSeed, uint64_t n)
{
    uint64_t seed = tempSeed;
    for(int i = 0; i < n; i++) seed = ((uint64_t) next(&seed, 32) << 32) + next(&seed, 32);
    return seed;
}

uint64_t getDatacellLength(FILE* chainFile)
{
    //save previous offset
    uint64_t savedOffset = ftell(chainFile);

    //now seek to the end of the file and find how many datacells it is
    fseek(chainFile, 0L, SEEK_END);
    uint64_t fileLength = ftell(chainFile)/sizeof(DataCell);

    //restore offset and return
    fseek(chainFile, fileLength, SEEK_SET);
    return fileLength;
}

void getNthCell(FILE* chainFile, DataCell* returnedData, uint64_t n)
{
    int error;

    //seek to nth datacell
    fseek(chainFile, n*2*sizeof(uint64_t), SEEK_SET);

    error = fread(&returnedData->input, sizeof(uint64_t), 1, chainFile);
    error = fread(&returnedData->hash, sizeof(uint64_t), 1, chainFile);
    returnedData->index = n;
}

uint64_t getNthHash(FILE* chainFile, uint64_t n)
{
    DataCell temp;
    getNthCell(chainFile, &temp, n);
    return temp.hash;
}

uint64_t getNthInput(FILE* chainFile, uint64_t n)
{
    DataCell temp;
    getNthCell(chainFile, &temp, n);
    return temp.input;
}

int isHashInDb(FILE* chainFile, DataCell* returnedData, uint64_t hash)
{
    //will be multiple instances of the same chain
    //returns 0 if not in file, 1 if in file
    //return the data in "returnedData"

    uint64_t lowBound = 0;
    uint64_t highBound = getDatacellLength(chainFile);

    uint64_t midpoint;
    DataCell midpointData;
    int error;
    DataCell previousCell;

    while(lowBound < highBound)
    {
        //get midpoint
        midpoint = lowBound + (highBound - lowBound)/2;

        //read datacell into midpointData
        getNthCell(chainFile, &midpointData, midpoint);

        if(midpointData.hash == hash)
        {
            //a chain was found, check to see if previous chain also matches

            //get previous chain
            getNthCell(chainFile, &previousCell, midpoint-1);
            while(previousCell.hash == midpointData.hash)
            {
                //messes stuff up, but whatever. we're returning right after this anyway
                midpoint--;
                getNthCell(chainFile, &previousCell, midpoint-1);
            }
            //get the chain at the new midpoint; the first occurance in the file
            getNthCell(chainFile, &midpointData, midpoint);
            *returnedData = midpointData;
            returnedData->index = midpoint;
            return 1;
        }
        else if(midpointData.hash < hash)
        {
            //guess was too low, move lowBound to midpoint
            //current midpoint also not valid
            lowBound = midpoint+1;
            //printf("too low\n");
        }
        else if(midpointData.hash > hash)
        {
            //guess was too high, move highBound to midpoint
            //current midpoint also not valid
            highBound = midpoint-1;
            //printf("too high\n");
        }
    }
    return 0;
}

__attribute__ ((const, always_inline, hot)) static inline uint64_t seedHash(uint64_t seed, uint64_t n)
{
    seed = nextLong(seed, n);
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

int findPreviousLink(uint64_t start, uint64_t targetHash, uint32_t reducerCount)
{
    //from a chain beginning, compute hashes until we find targetHash
    //then, print the targetHash and the seed which created the targetHash

    //outputs 1 if it finds a hash, 0 if not
    uint64_t previous;
    //while the next link is NOT the thing we're looking for, advance the chain
    for(uint64_t i = 0; i < CHAINLENGTH; i++)
    {

        previous = start;
        start = seedHash(start, reducerCount);

        //if next link is target hash, print and exit
        if(start == targetHash)
        {
            printf("wow, %ld --> %ld\n", nextLong(previous, reducerCount), targetHash);
            exit(0);
            return 1;
        }
        
    }
    return 0;
}

void* threadedLinker(void* structPointer)
{
    //make a new struct
    struct threadedParameters parameters;

    //set its pointer to the "structPointer" pointer
    parameters = *(struct threadedParameters*)structPointer;

    //interleave threads to iterate through the whole array
    for(uint64_t i = parameters.thread; i < parameters.chainCanidatesCount; i += THREADCOUNT)
    {
        findPreviousLink(parameters.chainCanidates[i], parameters.targetHash, parameters.reducerCount);
    }

    return NULL;
}

int main()
{
    FILE * table0 = fopen("/media/meox/NvmeSSD/table0Sorted.dat", "r");

    uint64_t targetHash = 0x7f5004a13a2c6506;
    uint64_t tempSeed = targetHash;
    uint32_t reducerCount = 1;


    //dynamically sized array stuff
    uint64_t chainCanidatesLength = 1;
    uint64_t chainCanidatesCount = 0;
    uint64_t* chainCanidates = (uint64_t*)malloc(chainCanidatesLength * sizeof(uint64_t));

    DataCell temp;
    tempSeed = temp.input;

    //add all possible chains to an array
    for(uint64_t i = 0; i < CHAINLENGTH; i++)
    {
        //get next hash
        tempSeed = seedHash(tempSeed, reducerCount);

        if(!isHashInDb(table0, &temp, tempSeed)) continue;

        //if the next hash in the table is the same, check that chain too
        while(temp.hash == tempSeed)
        {
            //printf("index:%.10lu - %.16lx --> %.16lx (%2.2f%% complete)\n", temp.index, temp.input, temp.hash, 100*((float)i/(float)CHAINLENGTH));
            
            //add all possible chains to a huge list
            chainCanidates[chainCanidatesCount] = temp.input;

            //we've added another entry to the list
            chainCanidatesCount++;

            printf("%lu stored chains!\n", chainCanidatesCount);
            //if the next value would overflow,
            if(chainCanidatesCount == chainCanidatesLength)
            {
                //multiply the array length by 2
                chainCanidatesLength *= 2;

                //and reallocate the list to be that length
                //(the speed is worth the extra heap usage)
                chainCanidates = (uint64_t*)realloc((void*)chainCanidates, chainCanidatesLength * sizeof(uint64_t));
            }
            getNthCell(table0, &temp, temp.index+1);
        }
    }

    printf("%lu stored chains!\n", chainCanidatesCount);
    
    //initalize an array of parameters, one for each thread
    struct threadedParameters parametersArray[THREADCOUNT];

    //build the parameters
    for(uint32_t i = 0; i < THREADCOUNT; i++)
    {
        parametersArray[i].chainCanidates = chainCanidates;
        parametersArray[i].chainCanidatesCount = chainCanidatesCount;
        parametersArray[i].targetHash = targetHash;
        parametersArray[i].reducerCount = reducerCount;
        parametersArray[i].thread = i;
    }

    int error;
    pthread_t thread_id;
    for(uint32_t i = 0; i < THREADCOUNT; i++)
    {
        error = pthread_create(&thread_id, NULL, threadedLinker, &parametersArray[i]);
        if (error != 0){
            perror("Thread creation failed");
            return 1;
        }
    }

    for(uint32_t i = 0; i < THREADCOUNT; i++)
    {
        error = pthread_join(thread_id, NULL);
        if (error != 0){
            perror("Thread creation failed");
            return 1;
        }
    }


    //findPreviousLink(temp.input, targetHash, reducerCount);
}
