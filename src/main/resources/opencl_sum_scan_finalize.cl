__kernel void finalizeScan(__global int * workGroupSums, __global int * otherSums) {

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0);
    const int localThreadId = get_local_id(0);

    const int globalOffset = groupId * workGroupSize;

    otherSums[globalOffset + localThreadId] = workGroupSums[groupId] + otherSums[globalOffset + localThreadId];
}