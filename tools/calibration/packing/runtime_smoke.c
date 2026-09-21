#include "llama.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) return 2;
    int threads = argc == 3 ? atoi(argv[2]) : 1;
    if (threads < 1 || threads > 32) return 2;
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), NULL};
    mp.devices = devices;
    mp.n_gpu_layers = 0; mp.use_mmap = false; mp.check_tensors = true;
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 32; cp.n_batch = 8; cp.n_ubatch = 8; cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.offload_kqv = false; cp.op_offload = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    llama_token tokens[] = {1, 4, 5};
    struct llama_batch batch = llama_batch_get_one(tokens, 3);
    if (llama_decode(ctx, batch) != 0) return 5;
    const float * logits = llama_get_logits(ctx);
    const int count = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (!logits || count <= 0) return 6;
    for (int i = 0; i < count; ++i) if (!isfinite(logits[i])) return 7;
    printf("CPU Q1 GGUF load+prefill passed: %d finite logits\n", count);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
