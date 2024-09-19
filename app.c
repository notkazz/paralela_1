#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <stdio.h>
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;

int main() {
    // Carrega o classificador Haar Cascade para detecção de faces
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        printf("Erro ao carregar o classificador de faces.\n");
        return -1;
    }

    // Diretório de imagens
    std::string folder_path = "/app/Imagens/";

    // Lista de arquivos de imagem a processar
    std::vector<std::string> image_files;

    // Itera sobre o diretório e coleta arquivos que não começam com "output_"
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string filename = entry.path().filename().string();
        if (entry.is_regular_file() && filename.find("output_") != 0) {
            image_files.push_back(entry.path().string());
        }
    }

    // Variáveis para acumular o tempo total e o tempo específico de processamento
    double processing_time = 0.0;
    double total_time = 0.0;

    // Medir o tempo total (para todas as operações)
    double start_total_time = omp_get_wtime();

    // Processa cada imagem encontrada
    for (const std::string& path : image_files) {
        // Carrega a imagem
        cv::Mat img = cv::imread(path);

        if (img.empty()) {
            printf("Erro ao carregar a imagem %s\n", path.c_str());
            continue;
        }

        // Vetor para armazenar as faces detectadas
        std::vector<cv::Rect> faces;

        // Medir o tempo de conversão para escala de cinza e detecção
        double start_process_time = omp_get_wtime();

        // Conversão para escala de cinza
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Detecção de faces
        face_cascade.detectMultiScale(
            gray, faces, 
            1.01,     // scaleFactor: quanto menor, mais lento
            15,       // minNeighbors: quanto maior, mais rigoroso e mais lento
            0,        // flags: Use 0 para a configuração padrão
            cv::Size(30, 30) // minSize: tamanho mínimo da face a ser detectada
        );

        double end_process_time = omp_get_wtime();
        processing_time += (end_process_time - start_process_time);

        // Desenha retângulos ao redor das faces detectadas
        for (size_t j = 0; j < faces.size(); j++) {
            cv::rectangle(img, faces[j], cv::Scalar(255, 0, 0), 2);
        }

        // Salva a imagem processada com faces detectadas
        std::string output_path = folder_path + "output_" + fs::path(path).filename().string();
        cv::imwrite(output_path, img);
    }

    // Medir o tempo total (incluindo carregamento, detecção e salvamento)
    double end_total_time = omp_get_wtime();
    total_time = end_total_time - start_total_time;

    // Exibir o tempo total de conversão para escala de cinza e detecção
    printf("Tempo total de conversão para escala de cinza e detecção de faces: %3.2f segundos\n", processing_time);

    // Exibir o tempo total de execução (incluindo todas as operações)
    printf("Tempo total de execução (incluindo todas as operações): %3.2f segundos\n", total_time);

    return 0;
}