extern "C" __global__
void predictBatch(const double* inputs, const double* w0, const double* w1, 
                  double* outputs, int numRadars) {
    // Thread index represents one radar in the batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRadars) return;

    // Local memory for hidden layer activations
    double hidden[24];
    const int INPUT_SIZE = 18;
    const int HIDDEN_SIZE = 24;
    const int OUTPUT_SIZE = 3;

    // 1. Input to Hidden Layer (Dot Product + Leaky ReLU)
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double sum = 0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += inputs[idx * INPUT_SIZE + i] * w0[i * HIDDEN_SIZE + j];
        }
        // Leaky ReLU (slope 0.1 as defined in game.go)
        hidden[j] = (sum > 0) ? sum : sum * 0.1;
    }

    // 2. Hidden to Output Layer (Dot Product + Tanh)
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        double sum = 0;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += hidden[i] * w1[i * OUTPUT_SIZE + j];
        }
        // Tanh activation (as defined in game.go)
        outputs[idx * OUTPUT_SIZE + j] = tanh(sum);
    }
}
