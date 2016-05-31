 __kernel void scanSum(__global int * input, __global int * output, __global int * metadata, __local int * cache) {

    const int I = metadata[0];
    const int n = metadata[1];

    const int thid = get_local_id(0);
    int outIdx = 0, inIdx = 1;

    cache[thid] = (thid > 0) ? input[thid - 1] : I;
    barrier(CLK_GLOBAL_MEM_FENCE);

    //printf("cache[%d] = %d\n", thid, cache[thid]);

    for (int offset = 1; offset < n; offset *= 2) {

        outIdx = 1 - outIdx;
        inIdx = 1 - inIdx;

        const int kOut = outIdx * n;
        const int kIn = inIdx * n;

        if (thid >= offset) {

            const int firstSummand = cache[kIn + thid];
            const int secondSummand = cache[kIn + thid - offset];

            cache[kOut + thid] = firstSummand + secondSummand;

            //printf("[T%d]: %d (cache[%d]) <- %d (cache[%d]) + %d (cache[%d])\n", thid, cache[kOut + thid], kOut + thid, firstSummand, kIn + thid, secondSummand, kIn + thid - offset);
        } else {
            cache[kOut + thid] = cache[kIn + thid];
            //printf("[T%d]: cache[%d] = %d <- cache[%d]\n", thid, kOut + thid, cache[kIn + thid], kIn + thid);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    //printf("[T%d]: output[%d] = %d\n", thid, thid, cache[thid]);
    output[thid] = cache[outIdx * n + thid];
}