#define DEBUG 1
#define TRACE 0

 __kernel void scanSum(__global int * input, __global int * output, __global int * workGroupSums, __local int * cache) {

    const int n = get_local_size(0) * 2;

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0) * 2;

    const int globalThreadId = get_global_id(0);
    const int localThreadId = get_local_id(0);

    const int globalOffset = groupId * workGroupSize;

    int outIdx = 0, inIdx = 1;

    cache[2 * localThreadId] = input[globalOffset + 2 * localThreadId];
    cache[2 * localThreadId + 1] = input[globalOffset + 2 * localThreadId + 1];
    barrier(CLK_GLOBAL_MEM_FENCE);

    #if DEBUG
    printf("%d: [GR%.2d] [GT%.2d] [LT%.2d]: cache[%d] = %d\n", __LINE__, groupId, globalThreadId, localThreadId, 2 * localThreadId, cache[2 * localThreadId]);
    printf("%d: [GR%.2d] [GT%.2d] [LT%.2d]: cache[%d] = %d\n", __LINE__, groupId, globalThreadId, localThreadId, 2 * localThreadId + 1, cache[2 * localThreadId + 1]);
    #endif

    int offset = 1;

    for (int depth = n >> 1; depth > 0; depth >>= 1) {

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (localThreadId < depth) {

            const int firstIndex = offset * ( 2 * localThreadId + 1 ) - 1;
            const int secondIndex = offset * ( 2 * localThreadId + 2 ) - 1;

            const int valueAtFirstIndex = cache[firstIndex];
            const int valueAtSecondIndex = cache[secondIndex];

            const int sum = valueAtFirstIndex + valueAtSecondIndex;

            cache[secondIndex] = sum;

            #if DEBUG
            printf("%d: [GR%.2d] [GT%.2d] [LT%.2d] [D%.2d]: cache[%d] (%d) <- cache[%d] (%d) + cache[%d] (%d) \n", __LINE__, groupId, globalThreadId, localThreadId, depth, secondIndex, sum, firstIndex, valueAtFirstIndex, secondIndex, valueAtSecondIndex);
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
        workGroupSums[groupId] = cache[n - 1];
        cache[2 * localThreadId + 2] = cache[n - 1];

        #if DEBUG
        printf("%d: [GR%.2d] [LT%.2d]: workGroupSums[%d] = %d\n", __LINE__, groupId, localThreadId, groupId, workGroupSums[groupId]);
        #endif
    }

    if (localThreadId == 0) {

        #if DEBUG
        printf("%d: [GR%.2d] [GT%.2d] [LT%.2d]: cache[%d] = 0\n", __LINE__, groupId, globalThreadId, localThreadId, n - 1);
        #endif

        cache[n - 1] = 0;
    }

    for (int depth = 1; depth < n; depth *= 2) {

        offset >>= 1;
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (localThreadId < depth) {
            const int firstIndex = offset * ( 2 * localThreadId + 1 ) - 1;
            const int secondIndex = offset * ( 2 * localThreadId + 2 ) - 1;

            const int valueAtFirstIndex = cache[firstIndex];
            const int valueAtSecondIndex = cache[secondIndex];
            const int sum = valueAtFirstIndex + valueAtSecondIndex;

            // swap
            cache[firstIndex] = valueAtSecondIndex;
            cache[secondIndex] = sum;

            #if DEBUG
            printf("%d: [GR%.2d] [GT%.2d] [LT%.2d] [D%.2d]: cache[%d] (%d) <- cache[%d] (%d) + cache[%d] (%d)\n", __LINE__, groupId, globalThreadId, localThreadId, depth, secondIndex, sum, firstIndex, valueAtFirstIndex, secondIndex, valueAtSecondIndex);
            #endif
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    const int firstIndexOfOutput = globalOffset + 2 * localThreadId;
    const int secondIndexOfOutput = globalOffset + 2 * localThreadId + 1;

    const int indexOfCacheValueToWriteToFirstIndexOfOutput = 2 * localThreadId + 1;
    const int indexOfCacheValueToWriteToSecondIndexOfOutput = 2 * localThreadId + 2;

    const int cacheValueToWriteToFirstIndex = cache[indexOfCacheValueToWriteToFirstIndexOfOutput];
    const int cacheValueToWriteToSecondIndex = cache[indexOfCacheValueToWriteToSecondIndexOfOutput];

    #if DEBUG
    printf("%d: [GR%.2d] [GT%.2d] [LT%.2d]: output[%d] = %d\n", __LINE__, groupId, globalThreadId, localThreadId, firstIndexOfOutput, cacheValueToWriteToFirstIndex);
    printf("%d: [GR%.2d] [GT%.2d] [LT%.2d]: output[%d] = %d\n", __LINE__, groupId, globalThreadId, localThreadId, secondIndexOfOutput, cacheValueToWriteToSecondIndex);
    #endif

    output[firstIndexOfOutput] = cacheValueToWriteToFirstIndex;
    output[secondIndexOfOutput] = cacheValueToWriteToSecondIndex;
}

__kernel void finalizeScan(__global int * workGroupSums, __global int * otherSums) {

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0) * 2;
    const int localThreadId = get_local_id(0);

    const int firstLocalThreadId = 2 * localThreadId;
    const int secondLocalThreadId = 2 * localThreadId + 1;

    const int globalOffset = groupId * workGroupSize;

    if (groupId > 0) {

        #if DEBUG
        printf("%d: [GR%.2d] [LT%.2d]: otherSums[%d] = otherSums[%d] (%d) + workGroupSums[%d] (%d)\n", __LINE__, groupId, localThreadId, globalOffset + localThreadId, globalOffset + localThreadId, otherSums[globalOffset + localThreadId], groupId - 1, workGroupSums[groupId - 1]);
        #endif

        otherSums[globalOffset + firstLocalThreadId] = otherSums[globalOffset + firstLocalThreadId] + workGroupSums[groupId - 1];
        otherSums[globalOffset + secondLocalThreadId] = otherSums[globalOffset + secondLocalThreadId] + workGroupSums[groupId - 1];
    } else {
        #if DEBUG
        printf("%d: [GR%.2d] [LT%.2d]: otherSums[%d] = %d\n", __LINE__, groupId, localThreadId, globalOffset + firstLocalThreadId, otherSums[globalOffset + firstLocalThreadId]);
        #endif
    }
}