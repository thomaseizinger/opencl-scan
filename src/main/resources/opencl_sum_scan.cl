#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define DEBUG 1
#define TRACE 1

 __kernel void scanSum(__global int * input, __global int * output, __global int * workGroupSums, __local int * cache) {

    const int n = get_local_size(0) * 2;

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0) * 2;

    const int globalThreadId = get_global_id(0);

    const int localThreadId = get_local_id(0);

    const int firstLocalThreadId = localThreadId;
    const int secondLocalThreadId = localThreadId + (n / 2);

    const int firstLocalThreadIdBankOffset = CONFLICT_FREE_OFFSET(firstLocalThreadId);
    const int secondLocalThreadIdBankOffset = CONFLICT_FREE_OFFSET(secondLocalThreadId);

    const int globalOffset = groupId * workGroupSize;

    int outIdx = 0, inIdx = 1;

    cache[firstLocalThreadId + firstLocalThreadIdBankOffset] = input[globalOffset + firstLocalThreadId];
    cache[secondLocalThreadId + secondLocalThreadIdBankOffset] = input[globalOffset + secondLocalThreadId];
    barrier(CLK_GLOBAL_MEM_FENCE);

    #if DEBUG
    printf("%d: [GR%.2d] [LT%.2d]: cache[%d] = %d\n", __LINE__, groupId, localThreadId, firstLocalThreadId + firstLocalThreadIdBankOffset, cache[firstLocalThreadId + firstLocalThreadIdBankOffset]);
    printf("%d: [GR%.2d] [LT%.2d]: cache[%d] = %d\n", __LINE__, groupId, localThreadId, secondLocalThreadId + secondLocalThreadIdBankOffset, cache[secondLocalThreadId + secondLocalThreadIdBankOffset]);
    #endif

    int offset = 1;

    for (int depth = n >> 1; depth > 0; depth >>= 1) {

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (localThreadId < depth) {

            int firstIndex = offset * ( 2 * localThreadId + 1 ) - 1;
            int secondIndex = offset * ( 2 * localThreadId + 1 + 1 ) - 1;

            firstIndex += firstLocalThreadIdBankOffset;
            secondIndex += secondLocalThreadIdBankOffset;

            const int valueAtFirstIndex = cache[firstIndex];
            const int valueAtSecondIndex = cache[secondIndex];

            const int sum = valueAtFirstIndex + valueAtSecondIndex;

            cache[secondIndex] = sum;

            #if DEBUG
            printf("%d: [GR%.2d] [LT%.2d] [D%.2d]: cache[%d] (%d) <- cache[%d] (%d) + cache[%d] (%d) \n", __LINE__, groupId, localThreadId, depth, secondIndex, sum, firstIndex, valueAtFirstIndex, secondIndex, valueAtSecondIndex);
            #endif

        }
        offset *= 2;
    }

    const bool lastThread = localThreadId == workGroupSize / 2 - 1;

    #if TRACE
    if (lastThread) {
        printf("%d: [GR%.2d] [LT%.2d]: lastThread = true, workGroupSize: %d\n", __LINE__, groupId, localThreadId, workGroupSize);
    } else {
        printf("%d: [GR%.2d] [LT%.2d]: lastThread = false, workGroupSize: %d\n", __LINE__, groupId, localThreadId, workGroupSize);
    }
    #endif

    if (lastThread) {

        const int offset = CONFLICT_FREE_OFFSET(n - 1);

        workGroupSums[groupId] = cache[n - 1 + offset];

        #if DEBUG
        printf("%d: [GR%.2d] [LT%.2d]: workGroupSums[%d] = %d\n", __LINE__, groupId, localThreadId, groupId, workGroupSums[groupId]);
        #endif
    }

    if (localThreadId == 0) {

        const int offset = CONFLICT_FREE_OFFSET(n - 1);

        #if DEBUG
        printf("%d: [GR%.2d] [LT%.2d]: cache[%d] = 0\n", __LINE__, groupId, localThreadId, n - 1 + offset);
        #endif

        cache[n - 1 + offset] = 0;
    }

    for (int depth = 1; depth < n; depth *= 2) {

        offset >>= 1;
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (localThreadId < depth) {

            int firstIndex = offset * ( 2 * localThreadId + 1 ) - 1;
            int secondIndex = offset * ( 2 * localThreadId + 1 + 1 ) - 1;

            firstIndex += firstLocalThreadIdBankOffset;
            secondIndex += secondLocalThreadIdBankOffset;

            #if DEBUG
            printf("%d: [LT%.2d] [O%.2d]: %.2d_tempValue = %d (%.2d_cache[%d])\n", __LINE__, localThreadId, offset, groupId, cache[firstIndex], groupId, firstIndex);
            #endif

            const int  tempValue = cache[firstIndex];

            #if DEBUG
            printf("%d:  [LT%.2d] [O%.2d]: %.2d_cache[%d] = %d (%.2d_cache[%d])\n", __LINE__, localThreadId, offset, groupId, firstIndex, cache[secondIndex], groupId, secondIndex);
            #endif

            cache[firstIndex] = cache[secondIndex];

            #if DEBUG
            printf("%d: [LT%.2d] [O%.2d]: %.2d_cache[%d] = %d (%.2d_cache[%d]) + %d (tempValue)\n", __LINE__, localThreadId, offset, groupId, secondIndex, cache[secondIndex], groupId, secondIndex, tempValue);
            #endif

            cache[secondIndex] += tempValue;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    const int firstIndexOfOutput = globalOffset + firstLocalThreadId;
    const int secondIndexOfOutput = globalOffset + secondLocalThreadId;

    const int indexOfCacheValueToWriteToFirstIndexOfOutput = firstLocalThreadId + 1 + firstLocalThreadIdBankOffset;
    const int indexOfCacheValueToWriteToSecondIndexOfOutput = secondLocalThreadId + 1 + secondLocalThreadIdBankOffset;

    const int cacheValueToWriteToFirstIndex = cache[indexOfCacheValueToWriteToFirstIndexOfOutput];
    const int cacheValueToWriteToSecondIndex = cache[indexOfCacheValueToWriteToSecondIndexOfOutput];

    #if DEBUG
    printf("%d: [GR%.2d] [LT%.2d]: output[%d] = %d (cache[%d])\n", __LINE__, groupId, localThreadId, firstIndexOfOutput, cacheValueToWriteToFirstIndex, indexOfCacheValueToWriteToFirstIndexOfOutput);
    printf("%d: [GR%.2d] [LT%.2d]: output[%d] = %d (cache[%d]\n", __LINE__, groupId, localThreadId, secondIndexOfOutput, cacheValueToWriteToSecondIndex, indexOfCacheValueToWriteToSecondIndexOfOutput);
    #endif

    output[firstIndexOfOutput] = cacheValueToWriteToFirstIndex;
    output[secondIndexOfOutput] = cacheValueToWriteToSecondIndex;
}

__kernel void finalizeScan(__global int * workGroupSums, __global int * otherSums) {

    const int globalThreadId = get_global_id(0);
    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0) * 2;
    const int localThreadId = get_local_id(0);

    const int firstLocalThreadId = 2 * localThreadId;
    const int secondLocalThreadId = 2 * localThreadId + 1;

    const int globalOffset = groupId * workGroupSize;

    if (groupId > 0) {

        #if TRACE
        printf("%d: [GR%.2d] [LT%.2d]: otherSums[%d] = otherSums[%d] (%d) + workGroupSums[%d] (%d)\n", __LINE__, groupId, localThreadId, globalOffset + firstLocalThreadId, globalOffset + firstLocalThreadId, otherSums[globalOffset + firstLocalThreadId], groupId - 1, workGroupSums[groupId - 1]);
        printf("%d: [GR%.2d] [LT%.2d]: otherSums[%d] = otherSums[%d] (%d) + workGroupSums[%d] (%d)\n", __LINE__, groupId, localThreadId, globalOffset + secondLocalThreadId, globalOffset + secondLocalThreadId, otherSums[globalOffset + secondLocalThreadId], groupId - 1, workGroupSums[groupId - 1]);
        #endif

        otherSums[globalOffset + firstLocalThreadId] = otherSums[globalOffset + firstLocalThreadId] + workGroupSums[groupId - 1];
        otherSums[globalOffset + secondLocalThreadId] = otherSums[globalOffset + secondLocalThreadId] + workGroupSums[groupId - 1];
    } else {
        #if TRACE
        printf("%d: [GR%.2d] [LT%.2d]: otherSums[%d] = %d\n", __LINE__, groupId, localThreadId, globalOffset + firstLocalThreadId, otherSums[globalOffset + firstLocalThreadId]);
        #endif
    }

    #if DEBUG
    printf("%d: [GR%.2d] [LT%.2d]: otherSums[%d] = %d\n", __LINE__, groupId, localThreadId, globalOffset + firstLocalThreadId, otherSums[globalOffset + firstLocalThreadId]);
    printf("%d: [GR%.2d] [LT%.2d]: otherSums[%d] = %d\n", __LINE__, groupId, localThreadId, globalOffset + secondLocalThreadId, otherSums[globalOffset + secondLocalThreadId]);
    #endif
}