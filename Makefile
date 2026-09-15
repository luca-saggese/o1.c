# HiDream GB10 inference engine — M1 native build

CC      ?= gcc
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra
CPPFLAGS += -Iinclude -Isrc/io

BIN      := build/hidream
TEST_BIN := build/test_model_loader
SRCS     := src/main.c src/model/model.c src/io/json.c
OBJS     := $(SRCS:.c=.o)
TEST_SRCS := tests/unit/test_model_loader.c src/model/model.c src/io/json.c
TEST_OBJS := $(TEST_SRCS:.c=.o)

all: $(BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJS) -lm

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lm

%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -rf build src/main.o src/model/model.o src/io/json.o

.PHONY: all clean
