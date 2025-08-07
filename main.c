#include "include/cten.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  

void* _cten_malloc(size_t size);

enum MemoryPoolIds {
    PoolId_Default = 0,
    PoolId_Model = 1,
    PoolId_Optimizer = 2,
};

typedef struct Model {
    Tensor weight_1, weight_2;
    Tensor bias_1, bias_2;
} Model;

Tensor Model_forward(Model* model, Tensor x) {
    x = nn_linear(x, model->weight_1, model->bias_1);
    x = nn_relu(x);
    x = nn_linear(x, model->weight_2, model->bias_2);
    return x;
}

void test_adam_optimizer() {
    printf("--- Testing Adam Optimizer ---\n");
    const float target_w1 = 14.6f;
    const float target_w2 = -8.7f;
    const int iterations = 300;
    const float learning_rate = 0.2f;

    TensorShape w_shape = {1, 2, 0, 0};
    Tensor w = Tensor_new(w_shape, true); 

    optim_adam* optimizer = optim_adam_new(1, &w, learning_rate, 0.9f, 0.999f, 1e-8f);

    for (int i = 1; i <= iterations; ++i) {
        optim_adam_zerograd(optimizer);

        float w1 = w.data->flex[0];
        float w2 = w.data->flex[1];

        // MSE (Uncomment this to test this loss function, and comment below loss function)
        float loss = (w1 - target_w1) * (w1 - target_w1) + (w2 - target_w2) * (w2 - target_w2);
        float grad1 = 2 * (w1 - target_w1);
        float grad2 = 2 * (w2 - target_w2);

        // MAE (Uncomment this to test this loss function, and comment above loss function)
        // float loss = fabsf(w1 - target_w1) + fabsf(w2 - target_w2);
        // float grad1 = (w1 > target_w1) ? 1.0f : -1.0f;
        // if (w1 == target_w1) grad1 = 0.0f;
        // float grad2 = (w2 > target_w2) ? 1.0f : -1.0f;
        // if (w2 == target_w2) grad2 = 0.0f;


        if (i % 100 == 0 || i == 1) {
            printf("Iter: %-3d | Loss: %-8.4f | ", i, loss);
            Tensor_print(w);
        }

        if (w.node->grad.data == NULL) {
            w.node->grad = Tensor_zeros(w.shape, false);
        }
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;

        optim_adam_step(optimizer);
    }

    printf("--------------------------------\n");
    printf("Adam Test Complete:\n");
    printf("Target values:          (%.4f, %.4f)\n", target_w1, target_w2);
    printf("Final values:           ");
    Tensor_print(w);
    printf("--------------------------------\n\n");
}

void test_rmsprop_optimizer() {
    printf("--- Testing RMSProp Optimizer ---\n");
    const float target_w1 = 14.6f;
    const float target_w2 = -8.7f;
    const int iterations = 300;
    const float learning_rate = 0.3f;

    TensorShape w_shape = {1, 2, 0, 0};
    Tensor w = Tensor_new(w_shape, true);

    optim_rmsprop* optimizer = optim_rmsprop_new(1, &w, learning_rate, 0.9f, 1e-8f);

    for (int i = 1; i <= iterations; ++i) {
        optim_rmsprop_zerograd(optimizer);

        float w1 = w.data->flex[0];
        float w2 = w.data->flex[1];

        // MSE (Uncomment this to test this loss function, and comment below loss function)
        float loss = (w1 - target_w1) * (w1 - target_w1) + (w2 - target_w2) * (w2 - target_w2);
        float grad1 = 2 * (w1 - target_w1);
        float grad2 = 2 * (w2 - target_w2);

        // MAE (Uncomment this to test this loss function, and comment above loss function)
        // float loss = fabsf(w1 - target_w1) + fabsf(w2 - target_w2);
        // float grad1 = (w1 > target_w1) ? 1.0f : -1.0f;
        // if (w1 == target_w1) grad1 = 0.0f;
        // float grad2 = (w2 > target_w2) ? 1.0f : -1.0f;
        // if (w2 == target_w2) grad2 = 0.0f;

        if (i % 100 == 0 || i == 1) {
            printf("Iter: %-3d | Loss: %-8.4f | ", i, loss);
            Tensor_print(w);
        }

        if (w.node->grad.data == NULL) {
            w.node->grad = Tensor_zeros(w.shape, false);
        }
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;
        
        optim_rmsprop_step(optimizer);
    }
    
    printf("--------------------------------\n");
    printf("RMSProp Test Complete:\n");
    printf("Target values:          (%.4f, %.4f)\n", target_w1, target_w2);
    printf("Final values:           ");
    Tensor_print(w);
    printf("--------------------------------\n\n");
}

void test_adagrad_optimizer() {
    printf("--- Testing AdaGrad Optimizer ---\n");
    const float target_w1 = 14.6f;
    const float target_w2 = -8.7f;
    const int iterations = 300;
    const float learning_rate = 0.8f;
    TensorShape w_shape = {1, 2, 0, 0};
    Tensor w = Tensor_new(w_shape, true);

    optim_adagrad* optimizer = optim_adagrad_new(1, &w, learning_rate, 1e-8f);

    for (int i = 1; i <= iterations; ++i) {
        optim_adagrad_zerograd(optimizer);

        float w1 = w.data->flex[0];
        float w2 = w.data->flex[1];

        // MSE (Uncomment this to test this loss function, and comment below loss function)
        float loss = (w1 - target_w1) * (w1 - target_w1) + (w2 - target_w2) * (w2 - target_w2);
        float grad1 = 2 * (w1 - target_w1);
        float grad2 = 2 * (w2 - target_w2);

        // MAE (Uncomment this to test this loss function, and comment above loss function)
        // float loss = fabsf(w1 - target_w1) + fabsf(w2 - target_w2);
        // float grad1 = (w1 > target_w1) ? 1.0f : -1.0f;
        // if (w1 == target_w1) grad1 = 0.0f;
        // float grad2 = (w2 > target_w2) ? 1.0f : -1.0f;
        // if (w2 == target_w2) grad2 = 0.0f;

        if (i % 100 == 0 || i == 1) {
            printf("Iter: %-3d | Loss: %-8.4f | ", i, loss);
            Tensor_print(w);
        }

        if (w.node->grad.data == NULL) {
            w.node->grad = Tensor_zeros(w.shape, false);
        }
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;
        
        optim_adagrad_step(optimizer);
    }

    printf("--------------------------------\n");
    printf("AdaGrad Test Complete:\n");
    printf("Target values:          (%.4f, %.4f)\n", target_w1, target_w2);
    printf("Final values:           ");
    Tensor_print(w);
    printf("--------------------------------\n\n");
}

int main() {
    cten_initilize();
    cten_begin_malloc(PoolId_Default);
    
    printf("Optimizer Tests\n");
    test_adam_optimizer();
    test_rmsprop_optimizer();
    test_adagrad_optimizer();

    cten_end_malloc();
    cten_finalize();
    return 0;
}