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

void test_adam_optimizer_with_enhancements() {
    printf("--- Testing Adam Optimizer with Weight Decay & Gradient Clipping ---\n");
    const float target_w1 = 14.6f;
    const float target_w2 = -8.7f;
    const int iterations = 300;
    const float learning_rate = 0.2f;
    const float weight_decay = 0.01f;  // L2 regularization
    const float max_grad_norm = 1.0f;  // Gradient norm clipping

    TensorShape w_shape = {1, 2, 0, 0};
    Tensor w = Tensor_new(w_shape, true);

    // Updated Adam constructor with weight decay
    optim_adam* optimizer = optim_adam_new(1, &w, learning_rate, 0.9f, 0.999f, 1e-8f, weight_decay);

    printf("Hyperparameters: LR=%.3f, Weight_Decay=%.4f, Max_Grad_Norm=%.1f\n",
           learning_rate,
           weight_decay,
           max_grad_norm);

    for(int i = 1; i <= iterations; ++i) {
        optim_adam_zerograd(optimizer);

        float w1 = w.data->flex[0];
        float w2 = w.data->flex[1];

        // MSE Loss
        float loss = (w1 - target_w1) * (w1 - target_w1) + (w2 - target_w2) * (w2 - target_w2);
        float grad1 = 2 * (w1 - target_w1);
        float grad2 = 2 * (w2 - target_w2);

        if(i % 100 == 0 || i == 1) {
            printf("Iter: %-3d | Loss: %-8.4f | ", i, loss);
            Tensor_print(w);
        }

        if(w.node->grad.data == NULL) { w.node->grad = Tensor_zeros(w.shape, false); }
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;

        // Apply gradient clipping before optimizer step
        cten_clip_grad_norm(&w, 1, max_grad_norm);

        optim_adam_step(optimizer);
    }

    printf("--------------------------------\n");
    printf("Adam Test with Enhancements Complete:\n");
    printf("Target values:          (%.4f, %.4f)\n", target_w1, target_w2);
    printf("Final values:           ");
    Tensor_print(w);
    printf("--------------------------------\n\n");
}

void test_rmsprop_optimizer_with_enhancements() {
    printf("--- Testing RMSProp Optimizer with Weight Decay & Value Clipping ---\n");
    const float target_w1 = 14.6f;
    const float target_w2 = -8.7f;
    const int iterations = 300;
    const float learning_rate = 0.3f;
    const float weight_decay = 0.005f;  // Smaller weight decay
    const float max_grad_value = 0.8f;  // Value-based clipping

    TensorShape w_shape = {1, 2, 0, 0};
    Tensor w = Tensor_new(w_shape, true);

    // Updated RMSProp constructor with weight decay
    optim_rmsprop* optimizer = optim_rmsprop_new(1, &w, learning_rate, 0.9f, 1e-8f, weight_decay);

    printf("Hyperparameters: LR=%.3f, Weight_Decay=%.4f, Max_Grad_Value=%.1f\n",
           learning_rate,
           weight_decay,
           max_grad_value);

    for(int i = 1; i <= iterations; ++i) {
        optim_rmsprop_zerograd(optimizer);

        float w1 = w.data->flex[0];
        float w2 = w.data->flex[1];

        // MSE Loss
        float loss = (w1 - target_w1) * (w1 - target_w1) + (w2 - target_w2) * (w2 - target_w2);
        float grad1 = 2 * (w1 - target_w1);
        float grad2 = 2 * (w2 - target_w2);

        if(i % 100 == 0 || i == 1) {
            printf("Iter: %-3d | Loss: %-8.4f | ", i, loss);
            Tensor_print(w);
        }

        if(w.node->grad.data == NULL) { w.node->grad = Tensor_zeros(w.shape, false); }
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;

        // Apply gradient value clipping (symmetric: [-0.8, +0.8])
        cten_clip_grad_value(&w, 1, max_grad_value);

        optim_rmsprop_step(optimizer);
    }

    printf("--------------------------------\n");
    printf("RMSProp Test with Enhancements Complete:\n");
    printf("Target values:          (%.4f, %.4f)\n", target_w1, target_w2);
    printf("Final values:           ");
    Tensor_print(w);
    printf("--------------------------------\n\n");
}

void test_adagrad_optimizer_with_enhancements() {
    printf("--- Testing AdaGrad Optimizer with Weight Decay & Range Clipping ---\n");
    const float target_w1 = 14.6f;
    const float target_w2 = -8.7f;
    const int iterations = 300;
    const float learning_rate = 0.8f;
    const float weight_decay = 0.008f;
    const float min_grad = -1.2f;  // Asymmetric clipping
    const float max_grad = 0.6f;

    TensorShape w_shape = {1, 2, 0, 0};
    Tensor w = Tensor_new(w_shape, true);

    // Updated AdaGrad constructor with weight decay
    optim_adagrad* optimizer = optim_adagrad_new(1, &w, learning_rate, 1e-8f, weight_decay);

    printf("Hyperparameters: LR=%.3f, Weight_Decay=%.4f, Grad_Range=[%.1f, %.1f]\n",
           learning_rate,
           weight_decay,
           min_grad,
           max_grad);

    for(int i = 1; i <= iterations; ++i) {
        optim_adagrad_zerograd(optimizer);

        float w1 = w.data->flex[0];
        float w2 = w.data->flex[1];

        // MSE Loss
        float loss = (w1 - target_w1) * (w1 - target_w1) + (w2 - target_w2) * (w2 - target_w2);
        float grad1 = 2 * (w1 - target_w1);
        float grad2 = 2 * (w2 - target_w2);

        if(i % 100 == 0 || i == 1) {
            printf("Iter: %-3d | Loss: %-8.4f | ", i, loss);
            Tensor_print(w);
        }

        if(w.node->grad.data == NULL) { w.node->grad = Tensor_zeros(w.shape, false); }
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;

        // Apply asymmetric gradient range clipping
        cten_clip_grad_value_range(&w, 1, min_grad, max_grad);

        optim_adagrad_step(optimizer);
    }

    printf("--------------------------------\n");
    printf("AdaGrad Test with Enhancements Complete:\n");
    printf("Target values:          (%.4f, %.4f)\n", target_w1, target_w2);
    printf("Final values:           ");
    Tensor_print(w);
    printf("--------------------------------\n\n");
}

void test_sgd_optimizer_with_enhancements() {
    printf("--- Testing SGD Optimizer with Weight Decay, Momentum & Gradient Clipping ---\n");
    const float target_w1 = 14.6f;
    const float target_w2 = -8.7f;
    const int iterations = 500;
    const float learning_rate = 0.01f;
    const float momentum = 0.9f;
    const float weight_decay = 0.001f;
    const float max_grad_norm = 2.0f;

    TensorShape w_shape = {1, 2, 0, 0};
    Tensor w = Tensor_new(w_shape, true);

    // SGD constructor with weight decay
    optim_sgd* optimizer = optim_sgd_new(1, &w, weight_decay);
    optim_sgd_config(optimizer, learning_rate, momentum);

    printf("Hyperparameters: LR=%.3f, Momentum=%.1f, Weight_Decay=%.4f, Max_Grad_Norm=%.1f\n",
           learning_rate,
           momentum,
           weight_decay,
           max_grad_norm);

    for(int i = 1; i <= iterations; ++i) {
        optim_sgd_zerograd(optimizer);

        float w1 = w.data->flex[0];
        float w2 = w.data->flex[1];

        // MSE Loss
        float loss = (w1 - target_w1) * (w1 - target_w1) + (w2 - target_w2) * (w2 - target_w2);
        float grad1 = 2 * (w1 - target_w1);
        float grad2 = 2 * (w2 - target_w2);

        if(i % 150 == 0 || i == 1) {
            printf("Iter: %-3d | Loss: %-8.4f | ", i, loss);
            Tensor_print(w);
        }

        if(w.node->grad.data == NULL) { w.node->grad = Tensor_zeros(w.shape, false); }
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;

        // Apply gradient norm clipping
        cten_clip_grad_norm(&w, 1, max_grad_norm);

        optim_sgd_step(optimizer);
    }

    printf("--------------------------------\n");
    printf("SGD Test with Enhancements Complete:\n");
    printf("Target values:          (%.4f, %.4f)\n", target_w1, target_w2);
    printf("Final values:           ");
    Tensor_print(w);
    printf("--------------------------------\n\n");
}

void test_gradient_clipping_methods() {
    printf("--- Testing Different Gradient Clipping Methods ---\n");
    const float target_w1 = 10.0f;
    const float target_w2 = -5.0f;
    const int iterations = 5;  // Just a few iterations to show clipping in action

    TensorShape w_shape = {1, 2, 0, 0};
    Tensor w = Tensor_new(w_shape, true);

    // Set initial values that will create large gradients
    w.data->flex[0] = 0.0f;  // Far from target
    w.data->flex[1] = 0.0f;

    printf("Initial values: w1=%.2f, w2=%.2f (targets: %.2f, %.2f)\n",
           w.data->flex[0],
           w.data->flex[1],
           target_w1,
           target_w2);
    printf("This will create large gradients to demonstrate clipping...\n\n");

    for(int i = 1; i <= iterations; ++i) {
        printf("=== Iteration %d ===\n", i);

        float w1 = w.data->flex[0];
        float w2 = w.data->flex[1];

        // Large gradients on purpose
        float grad1 = 4 * (w1 - target_w1);  // 2x larger than normal MSE
        float grad2 = 4 * (w2 - target_w2);

        printf("Original gradients: [%.4f, %.4f]\n", grad1, grad2);

        // Test different clipping methods
        if(w.node->grad.data == NULL) { w.node->grad = Tensor_zeros(w.shape, false); }

        // Test 1: Norm clipping
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;
        cten_clip_grad_norm(&w, 1, 2.0f);
        printf("After norm clipping (max=2.0): [%.4f, %.4f]\n",
               w.node->grad.data->flex[0],
               w.node->grad.data->flex[1]);

        // Test 2: Value clipping
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;
        cten_clip_grad_value(&w, 1, 1.5f);
        printf("After value clipping (±1.5): [%.4f, %.4f]\n",
               w.node->grad.data->flex[0],
               w.node->grad.data->flex[1]);

        // Test 3: Range clipping
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;
        cten_clip_grad_value_range(&w, 1, -3.0f, 1.0f);
        printf("After range clipping [-3.0, 1.0]: [%.4f, %.4f]\n",
               w.node->grad.data->flex[0],
               w.node->grad.data->flex[1]);

        // Test 4: Positive only clipping
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;
        cten_clip_grad_positive(&w, 1, 0.8f);
        printf("After positive clipping (max=0.8): [%.4f, %.4f]\n",
               w.node->grad.data->flex[0],
               w.node->grad.data->flex[1]);

        // Test 5: Negative only clipping
        w.node->grad.data->flex[0] = grad1;
        w.node->grad.data->flex[1] = grad2;
        cten_clip_grad_negative(&w, 1, -0.5f);
        printf("After negative clipping (min=-0.5): [%.4f, %.4f]\n",
               w.node->grad.data->flex[0],
               w.node->grad.data->flex[1]);

        printf("\n");

        // Update parameters with clipped gradients (using the last clipping method)
        w.data->flex[0] -= 0.1f * w.node->grad.data->flex[0];
        w.data->flex[1] -= 0.1f * w.node->grad.data->flex[1];
    }

    printf("--------------------------------\n");
    printf("Gradient Clipping Methods Test Complete\n");
    printf("Final values: [%.4f, %.4f]\n", w.data->flex[0], w.data->flex[1]);
    printf("--------------------------------\n\n");
}

int main() {
    cten_initilize();
    cten_begin_malloc(PoolId_Default);

    printf("Enhanced Optimizer Tests with Weight Decay & Gradient Clipping\n");
    printf("==============================================================\n\n");

    // Test all optimizers with enhancements
    test_adam_optimizer_with_enhancements();
    test_rmsprop_optimizer_with_enhancements();
    test_adagrad_optimizer_with_enhancements();
    test_sgd_optimizer_with_enhancements();

    // Demonstrate different gradient clipping methods
    test_gradient_clipping_methods();

    cten_end_malloc();
    cten_finalize();
    return 0;
}