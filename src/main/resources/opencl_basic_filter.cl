#define DEBUG 0

__kernel void filterGreaterThan(__global int * source, __global int * included, __global int * metadata) {

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0);
    const int globalOffset = groupId * workGroupSize;
    const int threshold = metadata[0];
    const int localThreadId = get_local_id(0);

    const int currentItemIndex = globalOffset + localThreadId;
    const int currentItem = source[currentItemIndex];

    #if DEBUG
        printf("currentItem = source[%d] (%d)\n", currentItemIndex, currentItem);
    #endif

    if (currentItem > threshold) {

        #if DEBUG
            printf("currentItem (%d) > threshold (%d) = true\n", currentItem, threshold);
        #endif

        included[currentItemIndex] = 1;
    } else {

        #if DEBUG
            printf("currentItem (%d) > threshold (%d) = false\n", currentItem, threshold);
        #endif

        included[currentItemIndex] = 0;
    }
}

__kernel void filterLessThan(__global int * source, __global int * included, __global int * metadata) {

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0);
    const int globalOffset = groupId * workGroupSize;
    const int threshold = metadata[0];
    const int localThreadId = get_local_id(0);

    const int currentItemIndex = globalOffset + localThreadId;
    const int currentItem = source[currentItemIndex];

    #if DEBUG
        printf("currentItem = source[%d] (%d)\n", currentItemIndex, currentItem);
    #endif

    if (currentItem < threshold) {

        #if DEBUG
            printf("currentItem (%d) < threshold (%d) = true\n", currentItem, threshold);
        #endif

        included[currentItemIndex] = 1;
    } else {

        #if DEBUG
            printf("currentItem (%d) < threshold (%d) = false\n", currentItem, threshold);
        #endif

        included[currentItemIndex] = 0;
    }
}

__kernel void filterEquals(__global int * source, __global int * included, __global int * metadata) {

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0);
    const int globalOffset = groupId * workGroupSize;
    const int threshold = metadata[0];
    const int localThreadId = get_local_id(0);

    const int currentItemIndex = globalOffset + localThreadId;
    const int currentItem = source[currentItemIndex];

    #if DEBUG
        printf("currentItem = source[%d] (%d)\n", currentItemIndex, currentItem);
    #endif

    if (currentItem == threshold) {

        #if DEBUG
            printf("currentItem (%d) == threshold (%d) = true\n", currentItem, threshold);
        #endif

        included[currentItemIndex] = 1;
    } else {

        #if DEBUG
            printf("currentItem (%d) == threshold (%d) = false\n", currentItem, threshold);
        #endif

        included[currentItemIndex] = 0;
    }
}

__kernel void applyFilter(__global int * source, __global int * dest, __global int * scannedAddresses) {

    const int groupId = get_group_id(0);
    const int workGroupSize = get_local_size(0);
    const int globalOffset = groupId * workGroupSize;
    const int localThreadId = get_local_id(0);
    const int currentItemIndex = globalOffset + localThreadId;

    const int targetAddress = scannedAddresses[currentItemIndex];
    const int previousAddress = scannedAddresses[currentItemIndex - 1];

    if (targetAddress != previousAddress) {
        #if DEBUG
            printf("dest[%d] = source[%d] (%d)\n", targetAddress, currentItemIndex, source[currentItemIndex]);
        #endif

        dest[targetAddress] = source[currentItemIndex];
    }
}