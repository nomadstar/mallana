#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

#include <clocale>
#include <cmath>
#include <string>
#include <vector>

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, /*parse_special*/ true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    const int n_passes = getenv("DIAG_TWO_PASS") ? atoi(getenv("DIAG_TWO_PASS")) : 1;

    for (int pass = 0; pass < n_passes; ++pass) {
        fprintf(stderr, "\n########## DIAG PASS %d ##########\n", pass);
        llama_memory_clear(llama_get_memory(ctx), true);

        int i_logits = tokens.size() - 1;
        const char * split_env = getenv("DIAG_SPLIT");
        const int n_split = split_env ? atoi(split_env) : 0;   // decode last n_split tokens one-by-one
        if (n_split > 0 && (int) tokens.size() > n_split) {
            const int n_prefix = tokens.size() - n_split;
            if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_prefix))) {
                LOG_ERR("%s : failed to eval prefix\n", __func__);
                return false;
            }
            for (int i = n_prefix; i < (int) tokens.size(); ++i) {
                if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, 1))) {
                    LOG_ERR("%s : failed to eval token %d\n", __func__, i);
                    return false;
                }
            }
            i_logits = 0; // last single-token decode has one output at index 0
        } else {
            if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
                LOG_ERR("%s : failed to eval\n", __func__);
                return false;
            }
        }

        const float * logits = llama_get_logits_ith(ctx, i_logits);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        // top-5 by simple scan
        std::vector<int> top(5, -1);
        std::vector<float> topv(5, -1e30f);
        double sum = 0.0, sumsq = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            const float v = logits[i];
            sum += v; sumsq += (double) v * v;
            for (int k = 0; k < 5; ++k) {
                if (v > topv[k]) {
                    for (int m = 4; m > k; --m) { topv[m] = topv[m-1]; top[m] = top[m-1]; }
                    topv[k] = v; top[k] = i;
                    break;
                }
            }
        }
        fprintf(stderr, "DIAG pass=%d logits: mean=%.4f l2=%.2f top5:", pass, sum / n_vocab, sqrt(sumsq));
        for (int k = 0; k < 5; ++k) {
            fprintf(stderr, " [%d]=%.4f '%s'", top[k], topv[k], common_token_to_piece(ctx, top[k]).c_str());
        }
        fprintf(stderr, "\n");
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    base_callback_data cb_data;

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    if (!getenv("DIAG_NO_CB")) {
        params.cb_eval = common_debug_cb_eval<false>;
        params.cb_eval_user_data = &cb_data;
    }
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
