#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main() {
    // Carrega o classificador Haar Cascade para detecção de faces
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        printf("Erro ao carregar o classificador de faces.\n");
        return -1;
    }

    // Carrega a imagem
    cv::Mat img = cv::imread("imagem.jpg");
    if (img.empty()) {
        printf("Erro ao carregar a imagem.\n");
        return -1;
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
    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(img, faces[i], cv::Scalar(255, 0, 0), 2);
    }

    // Exibe a imagem com as faces detectadas
    cv::imshow("Detecção de Faces", img);
    cv::waitKey(0);

    return 0;
}
