#define DEBUG 0
#define TRACE 0

 __kernel void scanSum(__global int * input, __global int * output, __global int * workGroupSums, __local int * cache) {

    const int n = get_local_size(0) * 2;

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0) * 2;

    const int globalThreadId = get_global_id(0);

    const int localThreadId = get_local_id(0);
    const int firstLocalThreadId = 2 * localThreadId;
    const int secondLocalThreadId = 2 * localThreadId + 1;

    const int globalOffset = groupId * workGroupSize;

    int outIdx = 0, inIdx = 1;

    cache[firstLocalThreadId] = input[globalOffset + firstLocalThreadId];
    cache[secondLocalThreadId] = input[globalOffset + secondLocalThreadId];
    barrier(CLK_GLOBAL_MEM_FENCE);

    #if DEBUG
    printf("%d: [GR%.2d] [GT%.2d] [LT%.2d]: cache[%d] = %d\n", __LINE__, groupId, globalThreadId, localThreadId, firstLocalThreadId, cache[firstLocalThreadId]);
    printf("%d: [GR%.2d] [GT%.2d] [LT%.2d]: cache[%d] = %d\n", __LINE__, groupId, globalThreadId, localThreadId, secondLocalThreadId, cache[secondLocalThreadId]);
    #endif

    int offset = 1;

    for (int depth = n >> 1; depth > 0; depth >>= 1) {

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (localThreadId < depth) {

            const int firstIndex = offset * ( firstLocalThreadId + 1 ) - 1;
            const int secondIndex = offset * ( secondLocalThreadId + 1 ) - 1;

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
            const int firstIndex = offset * ( firstLocalThreadId + 1 ) - 1;
            const int secondIndex = offset * ( secondLocalThreadId + 1 ) - 1;

            #if DEBUG
            printf("%d: [GT%.2d] [LT%.2d] [O%.2d]: %.2d_tempValue = %d (%.2d_cache[%d])\n", __LINE__, globalThreadId, localThreadId, offset, groupId, cache[firstIndex], groupId, firstIndex);
            #endif

            const int  tempValue = cache[firstIndex];

            #if DEBUG
            printf("%d: [GT%.2d] [LT%.2d] [O%.2d]: %.2d_cache[%d] = %d (%.2d_cache[%d])\n", __LINE__, globalThreadId, localThreadId, offset, groupId, firstIndex, cache[secondIndex], groupId, secondIndex);
            #endif

            cache[firstIndex] = cache[secondIndex];

            #if DEBUG
            printf("%d: [GT%.2d] [LT%.2d] [O%.2d]: %.2d_cache[%d] = %d (%.2d_cache[%d]) + %d (tempValue)\n", __LINE__, globalThreadId, localThreadId, offset, groupId, secondIndex, cache[secondIndex], groupId, secondIndex, tempValue);
            #endif

            cache[secondIndex] += tempValue;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    const int firstIndexOfOutput = globalOffset + firstLocalThreadId;
    const int secondIndexOfOutput = globalOffset + secondLocalThreadId;

    const int indexOfCacheValueToWriteToFirstIndexOfOutput = firstLocalThreadId + 1;
    const int indexOfCacheValueToWriteToSecondIndexOfOutput = secondLocalThreadId + 1;

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