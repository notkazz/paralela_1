#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <stdio.h>

int main() {
    // Carrega o classificador Haar Cascade para detecção de faces
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        printf("Erro ao carregar o classificador de faces.\n");
        return -1;
    }

    // Carrega uma imagem ou lista de imagens
    std::vector<cv::Mat> images;
    images.push_back(cv::imread("imagem1.jpg"));
    //images.push_back(cv::imread("imagem2.jpg"));
    // Adicionar mais imagens se necessário
    
    // Verifica se as imagens foram carregadas corretamente
    for (size_t i = 0; i < images.size(); i++) {
        if (images[i].empty()) {
            printf("Erro ao carregar a imagem %zu.\n", i);
            return -1;
        }
    }

    // Variáveis para medir o tempo
    double starttime, stoptime;

    // Início da medição de tempo para a parte paralelizada
    starttime = omp_get_wtime(); 

    // Conversão para escala de cinza e detecção de faces
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat gray;
        cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Vetor para armazenar as faces detectadas
        std::vector<cv::Rect> faces;

        // Detecção de faces
        face_cascade.detectMultiScale(gray, faces);

        // Desenha retângulos ao redor das faces detectadas
        for (size_t j = 0; j < faces.size(); j++) {
            cv::rectangle(images[i], faces[j], cv::Scalar(255, 0, 0), 2);
        }
    }

    // Fim da medição de tempo
    stoptime = omp_get_wtime();
    
    // Mostra o tempo de execução da parte paralela
    printf("Tempo de execução paralelo: %3.2f segundos\n", stoptime - starttime);

    // Exibe as imagens com as faces detectadas
    for (size_t i = 0; i < images.size(); i++) {
        cv::imshow("Detecção de Faces", images[i]);
    }
    cv::waitKey(0);

    return 0;
}