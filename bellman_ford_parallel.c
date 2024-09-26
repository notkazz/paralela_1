#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

#define INF INT_MAX  // Definimos o infinito como o maior valor de inteiro
#define V 50000       // Número de vértices no grafo

// Estrutura para representar uma aresta
struct Edge {
    int src, dest, weight;
};

// Função de Bellman-Ford paralelizada
void bellman_ford(struct Edge edges[], int E, int src) {
    int distance[V];

    // Inicializa as distâncias de todos os vértices como infinito
    for (int i = 0; i < V; i++) {
        distance[i] = INF;
    }
    distance[src] = 0;  // A distância da fonte para si mesma é 0

    // Relaxa todas as arestas V-1 vezes (passos do algoritmo Bellman-Ford)
    for (int i = 1; i <= V - 1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < E; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int weight = edges[j].weight;
            if (distance[u] != INF && distance[u] + weight < distance[v]) {
                distance[v] = distance[u] + weight;
            }
        }
    }

    // Verifica a presença de ciclos de peso negativo
    for (int i = 0; i < E; i++) {
        int u = edges[i].src;
        int v = edges[i].dest;
        int weight = edges[i].weight;
        if (distance[u] != INF && distance[u] + weight < distance[v]) {
            printf("O grafo contém um ciclo de peso negativo.\n");
            return;
        }
    }

    // Exibe as distâncias calculadas
    printf("Vértice\tDistância da origem\n");
    for (int i = 0; i < V; i++) {
        if (distance[i] == INF) {
            printf("%d\tINF\n", i);
        } else {
            printf("%d\t%d\n", i, distance[i]);
        }
    }
}

int main() {
    int num_threads, E;

    // Recebe o número de threads do usuário
    printf("Digite o número de threads a serem usadas: ");
    scanf("%d", &num_threads);
    omp_set_num_threads(num_threads);

    // Definimos um grafo com V vértices e E arestas
    E = V * 10;  // Defina um número arbitrário de arestas
    struct Edge* edges = malloc(E * sizeof(struct Edge));

    // Inicializa as arestas com valores aleatórios
    srand(42);
    for (int i = 0; i < E; i++) {
        edges[i].src = rand() % V;
        edges[i].dest = rand() % V;
        edges[i].weight = (rand() % 20) - 10;  // Pesos aleatórios entre -10 e 9
    }

    // Executa o algoritmo de Bellman-Ford a partir do vértice 0
    double start_time = omp_get_wtime();
    bellman_ford(edges, E, 0);
    double end_time = omp_get_wtime();

    printf("Tempo de execução: %f segundos\n", end_time - start_time);

    // Libera a memória alocada
    free(edges);

    return 0;
}
