#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring> // Para el manejo de argumentos de línea de comandos

const int INPUT_SIZE = 28 * 28; // Tamaño de la imagen MNIST (28x28 píxeles)
const int OUTPUT_SIZE = 10;     // 10 clases (dígitos del 0 al 9)

class Perceptron {
private:
    float learning_rate;
    float bias;

public:
    std::vector<float> weights;

    Perceptron(float lr) : learning_rate(lr), bias(1.0f) {
        // Inicializar los pesos con una distribución normalizada
        weights.resize(INPUT_SIZE + 1); // +1 para el sesgo
        srand(time(0)); // Semilla aleatoria
        for (int i = 0; i <= INPUT_SIZE; ++i) {
            weights[i] = 0.01f * ((float)rand() / RAND_MAX);
        }
    }

    // Función de activación (escalón unitario)
    int activation_function(float x) {
        return (x >= 0) ? 1 : 0;
    }

    // Entrenamiento del perceptrón
    void train(const std::vector<std::vector<float>>& training_data, const std::vector<std::vector<float>>& labels, int epochs, float& best_accuracy) {
        std::cout << "Comenzando el entrenamiento" << std::endl;
        std::vector<float> best_weights = weights;

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            for (size_t i = 0; i < training_data.size(); ++i) {
                // Calcular la salida
                float prediction = predict(training_data[i]);

                // Calcular el error
                std::vector<float> true_label = labels[i];
                float error = 0.0f;
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    error += true_label[j] - prediction;
                }
                
                // Actualizar los pesos
                if (error != 0.0f) {
                    for (size_t j = 0; j <= weights.size(); ++j) { // <= para incluir el sesgo
                        float input = (j == INPUT_SIZE) ? bias : training_data[i][j]; // Si j == INPUT_SIZE, usar el sesgo
                        weights[j] += learning_rate * error * input;
                    }
                }
            }

            // Calcular precisión y error cada 10 épocas
            if (epoch % 10 == 0) {
                float accuracy = evaluate(training_data, labels);
                std::cout << "Epoch " << epoch << " - Precisión: " << accuracy * 100.0f << "%" << std::endl;

                // Comparar la precisión actual con la mejor precisión registrada
                if (accuracy > best_accuracy) {
                    best_accuracy = accuracy;
                    best_weights = weights;
                }
            }
        }

        // Guardar los mejores pesos en un archivo CSV
        save_weights_to_csv("best_weights.csv", best_weights);
    }

    // Predicción
    float predict(const std::vector<float>& input) {
        float sum = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            sum += input[i] * weights[i];
        }
        sum += bias * weights[INPUT_SIZE]; // Añadir el sesgo
        return activation_function(sum);
    }

    // Evaluación de precisión
    float evaluate(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& labels) {
        int correct = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            float prediction = predict(data[i]);
            int predicted_label = round(prediction);
            int true_label = 0;
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                if (labels[i][j] == 1.0f) {
                    true_label = j;
                    break;
                }
            }
            if (predicted_label == true_label) {
                correct++;
            }
        }
        return (float)correct / data.size();
    }

    // Guardar pesos en un archivo CSV
    void save_weights_to_csv(const std::string& file_path, const std::vector<float>& weights_to_save) {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "No se pudo abrir el archivo: " << file_path << std::endl;
            return;
        }

        for (size_t i = 0; i < weights_to_save.size(); ++i) {
            file << weights_to_save[i] << ",";
        }
        file.close();
    }
};

// Función para cargar los pesos desde un archivo CSV
std::vector<float> load_weights_from_csv(const std::string& file_path) {
    std::vector<float> loaded_weights;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo: " << file_path << std::endl;
        return loaded_weights;
    }

    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        loaded_weights.push_back(std::stof(token));
    }
    file.close();
    return loaded_weights;
}

// Función para convertir las etiquetas a codificación one-hot
std::vector<float> one_hot_encode(int label) {
    std::vector<float> encoded(OUTPUT_SIZE, 0.0f); // 10 clases en total
    encoded[label] = 1.0f;
    return encoded;
}

// Modificar la carga de datos para aplicar la codificación one-hot a las etiquetas
void load_data_from_csv(const std::string& file_path, std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& labels) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo: " << file_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<float> row_data;
        std::vector<float> row_labels;
        bool is_label = true;
        while (std::getline(ss, token, ',')) {
            if (is_label) {
                int label = std::stoi(token);
                row_labels = one_hot_encode(label); // Convertir la etiqueta a one-hot
                is_label = false;
            } else {
                if(std::stof(token) > 1){
                    row_data.push_back(1); // Normalización de datos
                }else{
                     row_data.push_back(0); // Normalización de datos
                }
            }
        }
        data.push_back(row_data);
        labels.push_back(row_labels);
    }
}

int main(int argc, char* argv[]) {
    // Rutas de los archivos CSV de entrenamiento y evaluación
    std::string train_file_path = "mnist_train.csv";
    std::string test_file_path = "mnist_test.csv";
    std::string best_weights_file_path = "best_weights.csv"; // Archivo CSV para los mejores pesos

    bool train_mode = false; // Indicador para el modo de entrenamiento

    // Analizar argumentos de línea de comandos
    if (argc > 1 && std::strcmp(argv[1], "--train") == 0) {
        train_mode = true;
    }

    // Cargar los mejores pesos si existen
    std::vector<float> best_weights = load_weights_from_csv(best_weights_file_path);
    std::vector<std::vector<float>> train_data, train_labels, test_data, test_labels;
    load_data_from_csv(train_file_path, train_data, train_labels);
    load_data_from_csv(test_file_path, test_data, test_labels);
    
    // Crear un perceptrón monocapa con tasa de aprendizaje 0.01
    Perceptron p(0.01f);
    
    


    if (train_mode) {
        float best_accuracy = 0.0f;
        if (!best_weights.empty()) {
            // Calcular best_accuracy utilizando los mejores pesos
            p = Perceptron(0.01f);
            p.weights = best_weights;
            best_accuracy = p.evaluate(train_data, train_labels);
        }
        // Entrenar el perceptrón con los datos de entrenamiento
        std::cout<<"Mejore Precision Train es:"<<best_accuracy * 100.0f<<std::endl;
        p.train(train_data, train_labels, 10000, best_accuracy); // Entrenar durante 100 épocas
    } else {
        p = Perceptron(0.01f); // Reiniciar el perceptrón con una nueva instancia
        p.weights = best_weights;
    }

    // Evaluar el perceptrón con los datos de prueba utilizando los mejores pesos

    best_weights = load_weights_from_csv(best_weights_file_path);

    
    p.weights = best_weights;
    float test_accuracy = p.evaluate(test_data, test_labels);
    std::cout << "Precisión en datos de prueba con los mejores pesos: " << test_accuracy * 100.0f<< std::endl;

    return 0;
}
