#define DEBUG 0

 __kernel void scanSum(__global int * input, __global int * output, __global int * workGroupSums, __local int * cache) {

    const int n = get_local_size(0);

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0);

    const int globalThreadId = get_global_id(0);
    const int localThreadId = get_local_id(0);

    const int globalOffset = groupId * workGroupSize;

    int outIdx = 0, inIdx = 1;

    cache[localThreadId] = (globalThreadId == 0) ? 0 : input[globalOffset + localThreadId - 1];
    barrier(CLK_GLOBAL_MEM_FENCE);

    #if DEBUG
    printf("%d: [GR%.2d] [GT%.2d] [LT%.2d]: cache[%d] = %d\n", __LINE__, groupId, globalThreadId, localThreadId, localThreadId, cache[localThreadId]);
    #endif

    int kOut = 0;
    int kIn = 0;

    for (int offset = 1; offset < n; offset *= 2) {

        barrier(CLK_GLOBAL_MEM_FENCE);

        outIdx = 1 - outIdx;
        inIdx = 1 - inIdx;

        kOut = outIdx * n + localThreadId;
        kIn = inIdx * n + localThreadId;

        if (localThreadId >= offset) {

            const int firstSummand = cache[kIn];
            const int secondSummand = cache[kIn - offset];

            cache[kOut] = firstSummand + secondSummand;

            #if DEBUG
            printf("%d: [GR%.2d] [LT%.2d] [O%d]: %d (cache[%d]) <- %d (cache[%d]) + %d (cache[%d])\n", __LINE__, groupId, localThreadId, offset, cache[kOut], kOut, firstSummand, kIn, secondSummand, kIn - offset);
            #endif
        } else {
            cache[kOut] = cache[kIn];

            #if DEBUG
            printf("%d: [GR%.2d] [LT%.2d] [O%d]: cache[%d] = %d <- cache[%d]\n", __LINE__, groupId, localThreadId, offset, kOut, cache[kIn], kIn);
            #endif
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    int valueToStoreInOutputArray = cache[kOut];
    int indexForValueInOutputArray = globalOffset + localThreadId;

    #if DEBUG
    printf("%d: [GR%.2d] [LT%.2d]: output[%d] = %d (cache[%d])\n", __LINE__, groupId, localThreadId, indexForValueInOutputArray, valueToStoreInOutputArray, kOut);
    #endif

    output[indexForValueInOutputArray] = valueToStoreInOutputArray;

    if (localThreadId == workGroupSize - 1) {
        workGroupSums[groupId] = valueToStoreInOutputArray;

        #if DEBUG
        printf("%d: [GR%.2d] [LT%.2d]: workGroupSums[%d] = %d\n", __LINE__, groupId, localThreadId, groupId, workGroupSums[groupId]);
        #endif
    }
}

__kernel void finalizeScan(__global int * workGroupSums, __global int * otherSums) {

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0);
    const int localThreadId = get_local_id(0);

    const int globalOffset = groupId * workGroupSize;

    otherSums[globalOffset + localThreadId] = workGroupSums[groupId] + otherSums[globalOffset + localThreadId];
}