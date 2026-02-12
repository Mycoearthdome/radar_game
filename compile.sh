/usr/local/cuda-13/bin/nvcc -ptx radar.cu -o radar.ptx -arch=compute_86 -code=sm_86 #nvidia-smi --query-gpu=compute_cap --format=csv,noheader
CGO_CFLAGS="-I/etc/alternatives/cuda-13/include" \
CGO_LDFLAGS="-L/etc/alternatives/cuda-13/lib64 -lcuda" \
go build -tags cuda -o game
