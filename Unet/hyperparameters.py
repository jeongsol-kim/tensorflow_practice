IMG_SIZE = 512
PATCH_SIZE = 128
PATCH_STRIDE = 64
BATCH_SIZE = int((IMG_SIZE-PATCH_SIZE)/PATCH_STRIDE+1)*int((IMG_SIZE-PATCH_SIZE)/PATCH_STRIDE+1)