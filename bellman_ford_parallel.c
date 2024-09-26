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
    int *distance_current = malloc(V * sizeof(int));
    int *distance_next = malloc(V * sizeof(int));

    // Inicializa as distâncias
    for (int i = 0; i < V; i++) {
        distance_current[i] = INF;
        distance_next[i] = INF;
    }
    distance_current[src] = 0;

    int converged = 0;  // Flag para verificar convergência

    // Relaxa as arestas V-1 vezes
    for (int i = 1; i <= V - 1; i++) {
        converged = 1;  // Assume que convergiu

        #pragma omp parallel for
        for (int j = 0; j < V; j++) {
            distance_next[j] = distance_current[j];
        }

        #pragma omp parallel for reduction(&& : converged)
        for (int j = 0; j < E; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int weight = edges[j].weight;
            if (distance_current[u] != INF && distance_current[u] + weight < distance_next[v]) {
                distance_next[v] = distance_current[u] + weight;
                converged = 0;  // Houve atualização, ainda não convergiu
            }
        }

        // Troca os ponteiros dos arrays
        int *temp = distance_current;
        distance_current = distance_next;
        distance_next = temp;

        // Se não houve atualização, pode sair do loop
        if (converged) {
            break;
        }
    }

    // Verifica ciclos negativos
    for (int i = 0; i < E; i++) {
        int u = edges[i].src;
        int v = edges[i].dest;
        int weight = edges[i].weight;
        if (distance_current[u] != INF && distance_current[u] + weight < distance_current[v]) {
            printf("O grafo contém um ciclo de peso negativo.\n");
            free(distance_current);
            free(distance_next);
            return;
        }
    }

    // Exibe as distâncias calculadas
    printf("Vértice\tDistância da origem\n");
    for (int i = 0; i < V; i++) {
        if (distance_current[i] == INF) {
            printf("%d\tINF\n", i);
        } else {
            printf("%d\t%d\n", i, distance_current[i]);
        }
    }

    // Libera memória alocada
    free(distance_current);
    free(distance_next);
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

    // Caso tenha problema ao rodar com alto nivel de dados descomente: Libera a memória alocada
    free(edges);

    return 0;
}