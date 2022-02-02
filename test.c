#include "stdio.h"
#include "fast_gpu/fast_gpu.h"

int main()
{
     int status = 0;
    
    // Check gpu status
    status = GPU_GetDevInfo();
    if(status < 0)
        printf("No device will handle overlaps.\r\n");
    else   
        printf("overlaps are supported on the device.\r\n");
}