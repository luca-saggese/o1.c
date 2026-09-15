# HiDream GB10 inference engine — M1 native build

CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra
CPPFLAGS += -Iinclude -Isrc/io -Isrc/model -Isrc/cuda

CUDA_HOME  ?= /usr/local/cuda
CUDA_CPPFLAGS := -I$(CUDA_HOME)/include
CUDA_LDFLAGS  := -L$(CUDA_HOME)/lib64 -lcudart
CUBLAS_LDFLAGS := -lcublas -lcublasLt

CORE_SRCS := src/model/model.c src/io/json.c src/io/sha256.c src/io/safetensors.c

BIN        := build/hidream
TEST_BIN   := build/test_model_loader
TEST_W_BIN := build/test_weights
CUBIN      := build/obj/cuda
CUDA_OBJS  := $(CUBIN)/support.o $(CUBIN)/norm.o $(CUBIN)/act.o \
              $(CUBIN)/gemm.o $(CUBIN)/rope.o $(CUBIN)/attn.o
TEST_P_BIN := build/test_primitives
TEST_P_SRCS := tests/unit/test_primitives.c $(CORE_SRCS)
TEST_P_OBJS := $(TEST_P_SRCS:.c=.o)
TEST_TOK_BIN := build/test_tokenizer
TEST_TOK_SRCS := tests/unit/test_tokenizer.c src/model/tokenizer.c src/model/model.c
TEST_TOK_OBJS := $(TEST_TOK_SRCS:.c=.o)

SRCS      := src/main.c $(CORE_SRCS) src/model/weights.c
OBJS      := $(SRCS:.c=.o)

TEST_SRCS   := tests/unit/test_model_loader.c src/model/model.c src/io/json.c
TEST_OBJS   := $(TEST_SRCS:.c=.o)
TEST_W_SRCS := tests/unit/test_weights.c $(CORE_SRCS) src/model/weights.c
TEST_W_OBJS := $(TEST_W_SRCS:.c=.o)

all: $(BIN)

test: $(TEST_BIN) $(TEST_W_BIN) $(TEST_P_BIN) $(TEST_TOK_BIN)
	./$(TEST_BIN)
	./$(TEST_W_BIN)
	./$(TEST_P_BIN)
	./$(TEST_TOK_BIN)

test-primitives: $(TEST_P_BIN)

$(TEST_BIN): $(TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJS) -lm

$(TEST_W_BIN): $(TEST_W_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_W_OBJS) $(CUDA_LDFLAGS) -lm

$(TEST_P_BIN): $(TEST_P_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_P_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) -lm -lstdc++

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(CUDA_LDFLAGS) -lm

%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

$(CUBIN)/%.o: src/cuda/%.cu
	@mkdir -p $(CUBIN)
	$(NVCC) -arch=sm_121 -O2 -std=c++17 $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

clean:
	rm -rf build
	find src tests -name '*.o' -delete

.PHONY: all test test-primitives clean