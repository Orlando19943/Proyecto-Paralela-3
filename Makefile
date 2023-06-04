FLAGS = -lcudart -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CUDA_FLAGS = -std=c++11 -I /usr/local/include/opencv4

default: cmake-build/out
	./cmake-build/out ./images/runway.pgm

cmake-build/out: hough.cu common/pgm.cpp
	nvcc $(CUDA_FLAGS) hough.cu common/pgm.cpp -o cmake-build/out $(FLAGS) -arch=sm_86
