CC = gcc
CFLAGS = -shared -fPIC -O3 -mavx2 -mfma -march=native -funroll-loops -flto
LDFLAGS = -lm -lpthread
TARGET = libgann.so
SRC = gann.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) gann.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
