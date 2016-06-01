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

    printf("[%d] [%d] [%d]: cache[%d] = %d\n", groupId, globalThreadId, localThreadId, localThreadId, cache[localThreadId]);

    for (int offset = 1; offset < n; offset *= 2) {

        outIdx = 1 - outIdx;
        inIdx = 1 - inIdx;

        const int kOut = outIdx * n;
        const int kIn = inIdx * n;

        if (localThreadId >= offset) {

            const int firstSummand = cache[kIn + localThreadId];
            const int secondSummand = cache[kIn + localThreadId - offset];

            cache[kOut + localThreadId] = firstSummand + secondSummand;

            //printf("[T%d]: %d (cache[%d]) <- %d (cache[%d]) + %d (cache[%d])\n", localThreadId, cache[kOut + localThreadId], kOut + localThreadId, firstSummand, kIn + localThreadId, secondSummand, kIn + localThreadId - offset);
        } else {
            cache[kOut + localThreadId] = cache[kIn + localThreadId];
            //printf("[T%d]: cache[%d] = %d <- cache[%d]\n", localThreadId, kOut + localThreadId, cache[kIn + localThreadId], kIn + localThreadId);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    printf("[G%d] [T%d]: output[%d] = %d\n", groupId, localThreadId, localThreadId, cache[localThreadId]);
    //output[localThreadId] = cache[outIdx * n + localThreadId];

    output[globalOffset + localThreadId] = cache[localThreadId];

    if (localThreadId == workGroupSize - 1) {
        workGroupSums[groupId] = cache[localThreadId];
        printf("[G%d] [T%d]: workGroupSums[%d] = %d\n", groupId, localThreadId, groupId, workGroupSums[groupId]);
    }
}