#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "skiplist_seq.h"

// main
int main(int argc, char *argv[])
{

    int Inserts = atoi(argv[1]);
    int deletes = atoi(argv[2]);
    int ttlOps = atoi(argv[3]);
    int key = atoi(argv[4]);

    srand(time(NULL));

    SkipList sl;
    initSkipList(&sl);

    struct timeval st, ed;
    long second, m_second;
    double m_seconds;

    gettimeofday(&st, NULL);

    Inserts = (ttlOps * Inserts) / 100;
    deletes = (ttlOps * deletes) / 100;
    int Contains = ttlOps - (Inserts + deletes);

    for (int i = 0; i < Inserts; i++)
    {
        int insertKey = rand() % key;
        insert(&sl, insertKey);
    }

    for (int i = 0; i < Contains; i++)
    {
        int containskey = rand() % key;
        contains(&sl, containskey);
    }

    for (int i = 0; i < deletes; i++)
    {
        int deletekey = rand() % key;
        delete (&sl, deletekey);
    }

    gettimeofday(&ed, NULL);

    second = ed.tv_sec - st.tv_sec;
    m_second = ed.tv_usec - st.tv_usec;
    m_seconds = ((second) * 1000 + m_second / 1000.0) + 0.5;
    printf("Total time taken: %f milliseconds\n", m_seconds);

    return 0;
}
