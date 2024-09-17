#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// revisar para seguir os padroes vistos em aulas.
// carregar imagens antes e iniciar o timer dpois de carregar as imagens, apenas na parte paralela seguindo a lei de ahemed
int main() {
    // Carrega o classificador Haar Cascade para detecção de faces
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        printf("Erro ao carregar o classificador de faces.\n");
        return -1;
    }

    // carrega uma imagem ou lista de imagens    

    // Conversão para escala de cinza
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    // Vetor para armazenar as faces detectadas
    std::vector<cv::Rect> faces;

    // Dependendo da velocidade de deteccao, fazer deteccao de varias imagens. Com um objetivo de 15 seg de execucao sequencial no lab.
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
