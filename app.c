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

    // Carrega uma imagem
    std::string path = "/app/Imagens/1.PNG";
    cv::Mat img = cv::imread(path);

    if (img.empty()) {
        printf("Erro ao carregar a imagem %s\n", path.c_str());
        return -1;
    }

    // Medir o tempo apenas da parte paralelizável
    double starttime_parallel = omp_get_wtime();

    // Conversão para escala de cinza
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    // Vetor para armazenar as faces detectadas
    std::vector<cv::Rect> faces;

    // Detecção de faces (parte que pode ser paralelizada)
    face_cascade.detectMultiScale(gray, faces);

    double stoptime_parallel = omp_get_wtime();

    // Desenha retângulos ao redor das faces detectadas
    for (size_t j = 0; j < faces.size(); j++) {
        cv::rectangle(img, faces[j], cv::Scalar(255, 0, 0), 2);
    }

    // Salva a imagem processada com faces detectadas
    std::string output_path = "/app/Imagens/output_1.PNG";
    cv::imwrite(output_path, img);

    printf("Imagem %s processada e salva em %s\n", path.c_str(), output_path.c_str());

    // Tempo de execução da parte paralelizável
    printf("Tempo de execução da parte paralelizável: %3.2f segundos\n", stoptime_parallel - starttime_parallel);

    return 0;
}