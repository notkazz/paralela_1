#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <stdio.h>
#include <string>

int main() {
    // Carrega o classificador Haar Cascade para detecção de faces
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        printf("Erro ao carregar o classificador de faces.\n");
        return -1;
    }

    // Processamento das imagens
    double starttime = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 1; i <= 1; i++) {
        // Constrói o caminho para a imagem
        std::string path = "/app/Imagens/" + std::to_string(i) + ".PNG";
        cv::Mat img = cv::imread(path);

        if (img.empty()) {
            printf("Erro ao carregar a imagem %s\n", path.c_str());
            continue;
        }

        // Conversão para escala de cinza
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Vetor para armazenar as faces detectadas
        std::vector<cv::Rect> faces;

        // Detecção de faces
        face_cascade.detectMultiScale(gray, faces);

        // Desenha retângulos ao redor das faces detectadas
        for (size_t j = 0; j < faces.size(); j++) {
            cv::rectangle(img, faces[j], cv::Scalar(255, 0, 0), 2);
        }

        // Salva a imagem processada com faces detectadas
        std::string output_path = "/app/Imagens/output_" + std::to_string(i) + ".PNG";
        cv::imwrite(output_path, img);

        printf("Imagem %s processada e salva em %s\n", path.c_str(), output_path.c_str());
    }

    double stoptime = omp_get_wtime();
    printf("Tempo de execução paralelo: %3.2f segundos\n", stoptime - starttime);

    return 0;
}