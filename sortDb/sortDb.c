#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//bytes divided by sizeof(uint64_t) * 2
#define DATABASESIZE 20149436416 / (sizeof(uint64_t) * 2)

typedef struct{
    uint64_t input;
    uint64_t hash;
} DataCell;

int comparator (const void * p1, const void * p2)
{
    if((*(DataCell*)p1).hash < (*(DataCell*)p2).hash) return -1;
    if((*(DataCell*)p1).hash > (*(DataCell*)p2).hash) return 1;
    if((*(DataCell*)p1).hash == (*(DataCell*)p2).hash) return 0;
}

int main()
{
    //alloc mem on heap for data
    DataCell* chains = malloc(sizeof(DataCell) * DATABASESIZE);
    FILE* database = fopen("/media/meox/StorageHDD/rainbowTable/table_1/table1ToSort.dat", "r");

    printf("reading from disk\n");
    int error;
    for(uint64_t i = 0; i < DATABASESIZE; i++)
    {
        error = fread(&chains[i].input, sizeof(uint64_t), 1, database);
        error = fread(&chains[i].hash, sizeof(uint64_t), 1, database);
    }
    fclose(database);

    printf("sorting...\n");
    qsort(chains, DATABASESIZE, sizeof(DataCell), comparator);


    FILE* databaseSorted = fopen("/media/meox/StorageHDD/rainbowTable/table_1/table1Sorted.dat", "w");
    printf("writing to disk\n");
    for(uint64_t i = 0; i < DATABASESIZE; i++)
    {
        error = fwrite(&chains[i].input, sizeof(uint64_t), 1, databaseSorted);
        error = fwrite(&chains[i].hash, sizeof(uint64_t), 1, databaseSorted);
    }

    fclose(databaseSorted);
    free(chains);
}
