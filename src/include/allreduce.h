#ifndef ALLREDUCE_H
#define ALLREDUCE_H

#include <chrono>
#include <cmath>
#include <ctime>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#define DATA_SIZE 64 /* 262144 = 1MB    256 = 1KB*/
#define RUN_ROUND 1000
/* 1: MPI_ALLREDUCE; 2: mpiRing; 3: mpiButterfly; 4: mpiTreeReduction; else: MPI_REDUCE */
#define LOCAL_REDUCE_ALGO 0
/* 1: mpiBtreeBcast; 2: mpiPipelineBtreeBcast; else MPI_BCAST */
#define LOCAL_BCAST_ALGO 0
#define DATA_NUM 50

using namespace std;

/************************** FUNCTION DECLARATIONS **************************/
static uint64_t getHostHash(const char *);
static void getHostName(char *, int);
void getLocalRank(int, int, int *);
void GetBtree(int, int, int *, int *, int *, int *);
void GetDtree(int, int, int *, int *, int *, int *, int *, int *, int *, int *);
void GenerateColors(int, vector<vector<int>> &);
void split_world(int, int, MPI_Comm *, MPI_Comm *);
void mpiAllreduce(void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
void mpiButterfly(void *, void *, int, int, int, MPI_Datatype, MPI_Comm);
void mpiRing(void *, void *, int, int, int, MPI_Datatype, MPI_Comm);
void mpiBtreeBcast(void *, int, int, int, MPI_Datatype, MPI_Comm);
void mpiPipelineBtreeBcast(void *, int, int, int, MPI_Datatype, MPI_Comm);
void mpiDBtreeReduction(void *, int, int, int, MPI_Datatype, MPI_Comm);
void mpiTreeReduction(void *, void *, int, int, int, MPI_Datatype, MPI_Comm);
void mpiHierarchicalAllreduce(void *, void *, int, int, int, MPI_Datatype, MPI_Op, MPI_Comm,
                              MPI_Comm);

vector<int> data_sizes_resnet50 = {
    32,    64,    128,    256,    512,    1000,   1024,   2048,    4096,    9408,    16384,  32768,
    36864, 65536, 131072, 147456, 262144, 524288, 598924, 1048576, 2048000, 2097152, 2359296};

vector<int> data_sizes = data_sizes_resnet50;

#define PRINT(pr_cmd)                                                                              \
    do {                                                                                           \
        if (rank == 0)                                                                             \
            pr_cmd;                                                                                \
    } while (0)

#define MPICHECK(cmd)                                                                              \
    do {                                                                                           \
        int e = cmd;                                                                               \
        if (e != MPI_SUCCESS) {                                                                    \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define PERF_COUNTER(cmd)                                                                          \
    do {                                                                                           \
        double begin, end;                                                                         \
        for (int i = 0; i < RUN_ROUND; i++) {                                                      \
            begin = double(clock());                                                               \
            cmd;                                                                                   \
            end = double(clock());                                                                 \
            avg += (end - begin) / RUN_ROUND;                                                      \
        }                                                                                          \
    } while (0)

#define GET_BASE(T, base, offset, count) (void *)(&(((T *)base)[offset * count]))
#define TO_ARRAY(arr) ((float *)arr)
#define IGNORE MPI_STATUS_IGNORE

/************************** HELPER FUNCTIONS **************************/

static uint64_t getHostHash(const char *string)
{
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

static void getHostName(char *hostname, int maxlen)
{
    MPI_Get_processor_name(hostname, &maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

void getLocalRank(int rank, int size, int *localRank)
{
    *localRank = 0;
    uint64_t hostHashs[size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t),
                           MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < size; p++) {
        if (p == rank)
            break;
        if (hostHashs[p] == hostHashs[rank])
            (*localRank)++;
    }
}

void GetBtree(int rank, int size, int *u, int *d0, int *d1, int *parentChildType)
{
    int up, down0, down1;
    int bit;
    for (bit = 1; bit < size; bit <<= 1) {
        if (bit & rank)
            break;
    }

    if (rank == 0) {
        *u = -1;
        *d0 = -1;
        // Child rank is > 0 so it has to be our child 1, not 0.
        *d1 = size > 1 ? bit >> 1 : -1;
        return;
    }

    up = (rank ^ bit) | (bit << 1);
    // if smaller than the parent, we are his first child, otherwise we're his
    // second
    if (up >= size)
        up = (rank ^ bit);
    *parentChildType = (rank < up) ? 0 : 1;
    *u = up;

    int lowbit = bit >> 1;
    // down0 is always within bounds
    down0 = lowbit == 0 ? -1 : rank - lowbit;

    down1 = lowbit == 0 ? -1 : rank + lowbit;
    // Make sure down1 is within bounds
    while (down1 >= size) {
        down1 = lowbit == 0 ? -1 : rank + lowbit;
        lowbit >>= 1;
    }
    *d0 = down0;
    *d1 = down1;
}

void GetDtree(int rank, int size, int *s0, int *d0_0, int *d0_1, int *parentChildType0, int *s1,
              int *d1_0, int *d1_1, int *parentChildType1)
{
    // First tree ... use a btree
    GetBtree(rank, size, s0, d0_0, d0_1, parentChildType0);
    // Second tree ... mirror or shift
    if (size % 2 == 1) {
        // shift
        int shiftrank = (rank - 1 + size) % size;
        int u, d0, d1;
        GetBtree(shiftrank, size, &u, &d0, &d1, parentChildType1);
        *s1 = u == -1 ? -1 : (u + 1) % size;
        *d1_0 = d0 == -1 ? -1 : (d0 + 1) % size;
        *d1_1 = d1 == -1 ? -1 : (d1 + 1) % size;
    } else {
        // mirror
        int u, d0, d1;
        GetBtree(size - 1 - rank, size, &u, &d0, &d1, parentChildType1);
        *s1 = u == -1 ? -1 : size - 1 - u;
        *d1_0 = d0 == -1 ? -1 : size - 1 - d0;
        *d1_1 = d1 == -1 ? -1 : size - 1 - d1;
    }
}

void GenerateColors(int size, vector<vector<int>> &colors)
{
    int color = 1;
    for (int i = 0; i < size; i++) {
        for (int j = i; j < size; j++) {
            if (i == j) {
                colors[i][j] = -1;
                continue;
            }
            colors[i][j] = color;
            colors[j][i] = color;
            color++;
        }
    }
}

void split_world(int rank, int localrank, MPI_Comm *local, MPI_Comm *world_main)
{

    int local_color = rank - localrank;
    int world_color = localrank == 0 ? 1 : MPI_UNDEFINED;

    MPI_Comm_split(MPI_COMM_WORLD, local_color, rank, local);
    MPI_Comm_split(MPI_COMM_WORLD, world_color, rank, world_main);
}

/************************** WARPPED MPI FUNCTIONS **************************/

void mpiAllreduce(void *sendbuff, int count, MPI_Datatype dtype, MPI_Op op, MPI_Comm COMM_WORLD)
{
    MPI_Allreduce(MPI_IN_PLACE, sendbuff, count, dtype, op, COMM_WORLD);
}

/************************** USER COLLECTIVE FUNCTIONS **************************/

void mpiButterfly(void *sendbuff, void *tmp, int count, int rank, int size, MPI_Datatype dtype,
                  MPI_Comm COMM_WORLD)
{
    int level = log2(size);

    for (int i = 0; i < level; i++) {
        int commNode = rank ^ (1 << i);

        MPI_Sendrecv(sendbuff, count, dtype, commNode, 0, tmp, count, dtype, commNode, 0,
                     COMM_WORLD, IGNORE);
        for (int j = 0; j < count; j++) {
            TO_ARRAY(sendbuff)[j] = TO_ARRAY(tmp)[j];
        }
    }
}

void mpiRing(void *sendbuff, void *recvData, int count, int rank, int size, MPI_Datatype dtype,
             MPI_Comm COMM_WORLD)
{

    int sendTo = (rank + 1) % size;
    int recvFrom = (rank - 1 + size) % size;
    int count_in_ring = count / size;

    // Reduce-Scatter
    for (int i = 0; i < size - 1; i++) {
        void *sendBase = GET_BASE(float, sendbuff, ((rank - i + size) % size), count_in_ring);
        void *recvBase = GET_BASE(float, sendbuff, ((rank - 1 - i + size) % size), count_in_ring);

        MPI_Sendrecv(sendBase, count_in_ring, dtype, sendTo, 0, recvData, count_in_ring, dtype,
                     recvFrom, 0, COMM_WORLD, IGNORE);
        for (int j = 0; j < count_in_ring; j++) {
            TO_ARRAY(recvBase)[j] += TO_ARRAY(recvData)[j];
        }
    }

    // All-Gather
    for (int i = 0; i < size - 1; i++) {
        void *sendBase = GET_BASE(float, sendbuff, ((rank + 1 - i + size) % size), count_in_ring);
        void *recvBase = GET_BASE(float, sendbuff, ((rank - i + size) % size), count_in_ring);

        MPI_Sendrecv(sendBase, count_in_ring, dtype, sendTo, 0, recvData, count_in_ring, dtype,
                     recvFrom, 0, COMM_WORLD, IGNORE);
        for (int j = 0; j < count_in_ring; j++) {
            TO_ARRAY(recvBase)[j] = TO_ARRAY(recvData)[j];
        }
    }
}

/**************** BTREE FUNCTIONS ****************/

void mpiBtreeBcast(void *sendbuff, int count, int rank, int size, MPI_Datatype dtype,
                   MPI_Comm COMM_WORLD)
{
    int recvFrom, sendToL, sendToR, parentChildType;

    GetBtree(rank, size, &recvFrom, &sendToL, &sendToR, &parentChildType);

    if (recvFrom >= 0) {
        MPI_Recv(sendbuff, count, dtype, recvFrom, 0, COMM_WORLD, IGNORE);
    }
    if (sendToL >= 0) {
        MPI_Send(sendbuff, count, dtype, sendToL, 0, COMM_WORLD);
    }
    if (sendToR >= 0) {
        MPI_Send(sendbuff, count, dtype, sendToR, 0, COMM_WORLD);
    }
}

void mpiPipelineBtreeBcast(void *sendbuff, int count, int rank, int size, MPI_Datatype dtype,
                           MPI_Comm COMM_WORLD)
{
    int recvFrom, sendToL, sendToR, parentChildType;
    int countInTree = count / size;

    GetBtree(rank, size, &recvFrom, &sendToL, &sendToR, &parentChildType);

    for (int i = 0; i < size; i++) {
        void *sendBase = GET_BASE(float, sendbuff, i, countInTree);
        void *recvBase = GET_BASE(float, sendbuff, i, countInTree);
        if (recvFrom >= 0) {
            MPI_Recv(recvBase, countInTree, dtype, recvFrom, 0, COMM_WORLD, IGNORE);
        }
        if (sendToL >= 0) {
            MPI_Send(sendBase, countInTree, dtype, sendToL, 0, COMM_WORLD);
        }
        if (sendToR >= 0) {
            MPI_Send(sendBase, countInTree, dtype, sendToR, 0, COMM_WORLD);
        }
    }
}

void mpiDBtreeReduction(void *sendbuff, int count, int rank, int size, MPI_Datatype dtype,
                        MPI_Comm COMM_WORLD)
{

    int recvFrom0, sendToL0, sendToR0, parentChildType0;
    int recvFrom1, sendToL1, sendToR1, parentChildType1;

    int count_in_tree = count / size;

    GetDtree(rank, size, &recvFrom0, &sendToL0, &sendToR0, &parentChildType0, &recvFrom1, &sendToL1,
             &sendToR1, &parentChildType1);

    float *recvbuff = (float *)malloc(count * sizeof(float));
    float *sendBase = (float *)GET_BASE(float, sendbuff, recvFrom0, count_in_tree);

    for (int i = 0; i < log2(size); i++) {
        if (sendToL0 >= 0) {
            void *sendBase = GET_BASE(float, sendbuff, sendToL0, count_in_tree);
            MPI_Send(sendBase, count_in_tree, dtype, sendToL0, 0, COMM_WORLD);
        }
        if (sendToR0 >= 0) {
            void *sendBase = GET_BASE(float, sendbuff, sendToR0, count_in_tree);
            MPI_Send(sendBase, count_in_tree, dtype, sendToR0, 0, COMM_WORLD);
        }
        if (sendToL1 >= 0) {
            void *sendBase = GET_BASE(float, sendbuff, sendToL1, count_in_tree);
            MPI_Send(sendBase, count_in_tree, dtype, sendToL1, 0, COMM_WORLD);
        }
        if (sendToR1 >= 0) {
            void *sendBase = GET_BASE(float, sendbuff, sendToR1, count_in_tree);
            MPI_Send(sendBase, count_in_tree, dtype, sendToR1, 0, COMM_WORLD);
        }
        if (recvFrom0 >= 0) {
            MPI_Recv(recvbuff, count_in_tree, dtype, recvFrom0, 0, COMM_WORLD, IGNORE);
            float *recvBase = (float *)GET_BASE(float, sendbuff, recvFrom0, count_in_tree);
            for (int j = 0; j < count_in_tree; j++) {
                recvBase[j] += recvbuff[j];
            }
        }
        if (recvFrom1 >= 0) {
            MPI_Recv(recvbuff, count_in_tree, dtype, recvFrom1, 0, COMM_WORLD, IGNORE);
            float *recvBase = (float *)GET_BASE(float, sendbuff, recvFrom1, count_in_tree);
            for (int j = 0; j < count_in_tree; j++) {
                recvBase[j] += recvbuff[j];
            }
        }
    }
}

void mpiTreeReduction(void *sendbuff, void *recvbuff, int count, int rank, int size,
                      MPI_Datatype dtype, MPI_Comm COMM_WORLD)
{
    int level = (int)log2(size);
    for (int i = 0; i < level; i++) {
        int commNode = rank ^ (1 << i);
        if (rank < commNode) {
            MPI_Recv(recvbuff, count, dtype, commNode, 0, COMM_WORLD, IGNORE);
            for (int j = 0; j < count; j++) {
                TO_ARRAY(sendbuff)[j] += TO_ARRAY(recvbuff)[j];
            }
        } else {
            MPI_Send(sendbuff, count, dtype, commNode, 0, COMM_WORLD);
        }
    }
}

/**************** HIERARCHICAL FUNCTIONS ****************/

void mpiHierarchicalAllreduce(void *sendbuff, void *recvbuff, int count, int rank, int size,
                              MPI_Datatype dtype, MPI_Op op, MPI_Comm COMM_LOCAL,
                              MPI_Comm COMM_WORLD_MAIN)
{
    int localRank, localSize;
    MPI_Comm_rank(COMM_LOCAL, &localRank);
    MPI_Comm_size(COMM_LOCAL, &localSize);

#if LOCAL_REDUCE_ALGO == 1
    /* Local Allreduce */
    MPI_Allreduce(MPI_IN_PLACE, sendbuff, count, dtype, op, COMM_LOCAL);
#elif LOCAL_REDUCE_ALGO == 2
    /* Local Ring */
    mpiRing(sendbuff, recvbuff, count, localRank, localSize, dtype, COMM_LOCAL);
#elif LOCAL_REDUCE_ALGO == 3
    /* Local mpiButterfly */
    mpiButterfly(sendbuff, recvbuff, count, localRank, localSize, dtype, COMM_LOCAL);
#elif LOCAL_REDUCE_ALGO == 4
    /* Local Tree Reduction */
    mpiTreeReduction(sendbuff, recvbuff, count, localRank, localSize, dtype, COMM_LOCAL);
#else
    /* Local Reduce */
    if (localRank == 0)
        MPI_Reduce(MPI_IN_PLACE, sendbuff, count, dtype, op, 0, COMM_LOCAL);
    else
        MPI_Reduce(sendbuff, NULL, count, dtype, op, 0, COMM_LOCAL);
#endif

    /* Node Allreduce */
    if (localRank == 0)
        MPI_Allreduce(MPI_IN_PLACE, sendbuff, count, dtype, op, COMM_WORLD_MAIN);

#if LOCAL_BCAST_ALGO == 1
    /* Btree Broadcast */
    mpiBtreeBcast(sendbuff, count, localRank, localSize, dtype, COMM_LOCAL);
#elif LOCAL_BCAST_ALGO == 2
    /* Pipelined Btree Broadcast */
    mpiPipelineBtreeBcast(sendbuff, count, localRank, localSize, dtype, COMM_LOCAL);
#else
    /* Local Broadcast */
    MPI_Bcast(sendbuff, count, dtype, 0, COMM_LOCAL);
#endif
}

#endif