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
              $(CUBIN)/gemm.o $(CUBIN)/rope.o $(CUBIN)/attn.o \
              $(CUBIN)/residual.o $(CUBIN)/embed.o $(CUBIN)/sched.o
TEST_P_BIN := build/test_primitives
TEST_P_SRCS := tests/unit/test_primitives.c $(CORE_SRCS)
TEST_P_OBJS := $(TEST_P_SRCS:.c=.o)
TEST_TOK_BIN := build/test_tokenizer
TEST_TOK_SRCS := tests/unit/test_tokenizer.c src/model/tokenizer.c src/model/model.c src/io/json.c
TEST_TOK_OBJS := $(TEST_TOK_SRCS:.c=.o)

TEST_BLOCK_BIN := build/test_block
TEST_BLOCK_SRCS := tests/unit/test_block.c src/model/block.c $(CORE_SRCS) src/model/weights.c
TEST_BLOCK_OBJS := $(TEST_BLOCK_SRCS:.c=.o)

TEST_FF_BIN := build/test_full_forward
TEST_FF_SRCS := tests/unit/test_full_forward.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
TEST_FF_OBJS := $(TEST_FF_SRCS:.c=.o)

TEST_TAIL_BIN := build/test_block_tail
TEST_TAIL_SRCS := tests/unit/test_block_tail.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
TEST_TAIL_OBJS := $(TEST_TAIL_SRCS:.c=.o)

TEST_M15_BIN := build/test_m1_5_scheduler
TEST_M15_SRCS := tests/unit/test_m1_5_scheduler.c src/model/block.c src/model/forward.c src/model/scheduler.c $(CORE_SRCS) src/model/weights.c
TEST_M15_OBJS := $(TEST_M15_SRCS:.c=.o)

SRCS      := src/main.c $(CORE_SRCS) src/model/weights.c src/model/block.c src/model/forward.c
OBJS      := $(SRCS:.c=.o)

TEST_SRCS   := tests/unit/test_model_loader.c src/model/model.c src/io/json.c
TEST_OBJS   := $(TEST_SRCS:.c=.o)
TEST_W_SRCS := tests/unit/test_weights.c $(CORE_SRCS) src/model/weights.c
TEST_W_OBJS := $(TEST_W_SRCS:.c=.o)

all: $(BIN)

test: $(TEST_BIN) $(TEST_W_BIN) $(TEST_P_BIN) $(TEST_TOK_BIN) $(TEST_BLOCK_BIN) $(TEST_FF_BIN) $(TEST_TAIL_BIN)
	./$(TEST_BIN)
	./$(TEST_W_BIN)
	./$(TEST_P_BIN)
	./$(TEST_TOK_BIN)
	./$(TEST_BLOCK_BIN)
	./$(TEST_FF_BIN)

test-primitives: $(TEST_P_BIN)

test-block: $(TEST_BLOCK_BIN)

test-full-forward: $(TEST_FF_BIN)
	./$(TEST_FF_BIN)

test-m15: $(TEST_M15_BIN)
	./$(TEST_M15_BIN)

test-tokenizer: $(TEST_TOK_BIN)
	./$(TEST_TOK_BIN)

$(TEST_BIN): $(TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJS) -lm

$(TEST_W_BIN): $(TEST_W_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_W_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) -lm -lstdc++

$(TEST_P_BIN): $(TEST_P_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_P_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) -lm -lstdc++

$(TEST_TOK_BIN): $(TEST_TOK_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_TOK_OBJS) -lm

$(TEST_BLOCK_BIN): $(TEST_BLOCK_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_BLOCK_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) -lm -lstdc++

$(TEST_FF_BIN): $(TEST_FF_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_FF_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) -lm -lstdc++

$(TEST_TAIL_BIN): $(TEST_TAIL_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_TAIL_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) -lm -lstdc++

$(TEST_M15_BIN): $(TEST_M15_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_M15_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) -lm -lstdc++

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) -lm -lstdc++

%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

$(CUBIN)/%.o: src/cuda/%.cu
	@mkdir -p $(CUBIN)
	$(NVCC) -arch=sm_121 -O2 -std=c++17 $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

clean:
	rm -rf build
	find src tests -name '*.o' -delete

.PHONY: all test test-primitives test-block test-full-forward test-tokenizer test-m15 clean