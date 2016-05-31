#define SWAP(a,b) {__local float *tmp=a;a=b;b=tmp;}

__kernel void sumScan(__global int *a, __global int *r, __local int *b, __local int *c) {

}
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint gs = get_local_size(0);
 
    c[lid] = b[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint s = 1; s < gs; s <<= 1) {
        if(lid > (s-1)) {
            c[lid] = b[lid]+b[lid-s];
        } else {
            c[lid] = b[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(b,c);
    }

    r[gid] = b[lid];
}