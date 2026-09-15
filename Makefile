# HiDream GB10 inference engine — M1 native build

CC      ?= gcc
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra
CPPFLAGS += -Iinclude -Isrc/io -Isrc/model

CUDA_HOME ?= /usr/local/cuda
CUDA_CPPFLAGS := -I$(CUDA_HOME)/include
CUDA_LDFLAGS  := -L$(CUDA_HOME)/lib64 -lcudart

BIN        := build/hidream
TEST_BIN   := build/test_model_loader
TEST_W_BIN := build/test_weights

CORE_SRCS := src/model/model.c src/io/json.c src/io/sha256.c src/io/safetensors.c
SRCS      := src/main.c $(CORE_SRCS) src/model/weights.c
OBJS      := $(SRCS:.c=.o)

TEST_SRCS   := tests/unit/test_model_loader.c src/model/model.c src/io/json.c
TEST_OBJS   := $(TEST_SRCS:.c=.o)
TEST_W_SRCS := tests/unit/test_weights.c $(CORE_SRCS) src/model/weights.c
TEST_W_OBJS := $(TEST_W_SRCS:.c=.o)

all: $(BIN)

test: $(TEST_BIN) $(TEST_W_BIN)
	./$(TEST_BIN)
	./$(TEST_W_BIN)

$(TEST_BIN): $(TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJS) -lm

$(TEST_W_BIN): $(TEST_W_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_W_OBJS) $(CUDA_LDFLAGS) -lm

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(CUDA_LDFLAGS) -lm

%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

clean:
	rm -rf build
	find src tests -name '*.o' -delete

.PHONY: all test clean