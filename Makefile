.PHONY: all
all: test

test: skiplist_seq.o test.o
	gcc skiplist_seq.o test.o -o test -std=c99

skiplist_seq.o: skiplist_seq.c skiplist_seq.h
	gcc -c skiplist_seq.c -o skiplist_seq.o -std=c99

test.o: test.c skiplist_seq.h
	gcc -c test.c -o test.o -std=c99

.PHONY: clean
clean:
	rm -f *.o test test_cuda