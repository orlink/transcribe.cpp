// transcribe-model.cpp - out-of-line definitions for the model and
// context base classes. Anchors the virtual destructors in one TU
// (avoids -Wweak-vtables) and holds set_languages().

#include "transcribe-model.h"

#include "transcribe-backend.h"
#include "transcribe-env.h"
#include "transcribe-session.h"

#include <cstdlib>
#include <utility>

transcribe_model::~transcribe_model() = default;

// Scheduler first, then the compute context (release_compute_scratch keeps
// that order). Runs after the derived destructors; see the header note.
transcribe::SessionCore::~SessionCore() {
    transcribe::release_compute_scratch(sched, compute_ctx);
}

// TRANSCRIBE_KEEP_SCRATCH_MB=<n>: keep a scheduler whose buffers total at
// most n MiB across offline runs (reset, not freed), so short runs do not
// rebuild and re-fault the same workspace every time. Unset or 0: always free.
static size_t keep_scratch_limit_bytes() noexcept {
    static const size_t limit = [] {
        const char * v = transcribe::env::str("TRANSCRIBE_KEEP_SCRATCH_MB");
        if (v == nullptr) {
            return static_cast<size_t>(0);
        }
        const long long mb = std::atoll(v);
        return mb > 0 ? static_cast<size_t>(mb) * 1024 * 1024 : static_cast<size_t>(0);
    }();
    return limit;
}

static size_t sched_buffer_bytes(ggml_backend_sched_t sched) noexcept {
    size_t total = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        total += ggml_backend_sched_get_buffer_size(sched, ggml_backend_sched_get_backend(sched, i));
    }
    return total;
}

void transcribe::SessionCore::release_scratch() noexcept {
    const size_t limit = keep_scratch_limit_bytes();
    if (sched != nullptr && limit > 0 && sched_buffer_bytes(sched) <= limit) {
        // The reset drops the scheduler's references to this run's graph,
        // whose tensors live in the compute context freed below.
        ggml_backend_sched_reset(sched);
        if (compute_ctx != nullptr) {
            ggml_free(compute_ctx);
            compute_ctx = nullptr;
        }
    } else {
        transcribe::release_compute_scratch(sched, compute_ctx);
    }
    on_scratch_released();
}

transcribe_session::~transcribe_session() = default;

void transcribe_session::clear_result() {
    tokens.clear();
    words.clear();
    segments.clear();
    speaker_segments.clear();
    full_text.clear();
    raw_text.clear();
    detected_language.clear();
    result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    has_result  = false;

    // Stream snapshot — lifecycle (stream_state) is NOT touched here;
    // the streaming dispatcher manages IDLE/ACTIVE/FINISHED/FAILED
    // explicitly. Everything else is per-utterance bookkeeping that
    // belongs with the result it describes.
    stream_revision                  = 0;
    n_committed_segments             = 0;
    n_committed_words                = 0;
    n_committed_tokens               = 0;
    stream_last_status               = TRANSCRIBE_OK;
    stream_audio_input_us            = 0;
    stream_audio_committed_us        = 0;
    stream_commit_policy             = TRANSCRIBE_STREAM_COMMIT_AUTO;
    stream_stable_prefix_agreement_n = 0;
    stream_committed_text.clear();
    stream_tentative_text.clear();
    stream_raw_tentative_start_bytes = 0;
    stream_raw_history.clear();
}

void transcribe_model::set_languages(std::vector<std::string> langs) {
    // Move the strings into the model so their c_str() pointers stay valid
    // for the model's lifetime. The pointer vector is rebuilt below because
    // the previous entries dangle after the storage is replaced.
    language_storage_ = std::move(langs);

    language_ptrs_.clear();
    language_ptrs_.reserve(language_storage_.size());
    for (const auto & s : language_storage_) {
        language_ptrs_.push_back(s.c_str());
    }

    // Publish count and pointer from the same backing storage. An empty
    // vector exposes (0, nullptr) rather than (0, &empty[0]).
    caps.n_languages = static_cast<int>(language_storage_.size());
    caps.languages   = language_ptrs_.empty() ? nullptr : language_ptrs_.data();
}

void transcribe_model::set_translate_target_languages(std::vector<std::string> langs) {
    // Same discipline as set_languages(): move strings into the model,
    // rebuild the pointer vector, then publish count + pointer together.
    translate_target_storage_ = std::move(langs);

    translate_target_ptrs_.clear();
    translate_target_ptrs_.reserve(translate_target_storage_.size());
    for (const auto & s : translate_target_storage_) {
        translate_target_ptrs_.push_back(s.c_str());
    }

    caps.n_translate_target_languages = static_cast<int>(translate_target_storage_.size());
    caps.translate_target_languages   = translate_target_ptrs_.empty() ? nullptr : translate_target_ptrs_.data();
}

void transcribe_model::set_translation_pairs(std::vector<std::string> pairs) {
    translation_pair_storage_ = std::move(pairs);
}

bool transcribe_model::allows_translation_pair(const char * src, const char * dst) const {
    if (translation_pair_storage_.empty()) {
        return true;
    }
    if (src == nullptr || dst == nullptr || src[0] == '\0' || dst[0] == '\0') {
        return false;
    }
    const std::string pair = std::string(src) + ">" + dst;
    for (const auto & allowed : translation_pair_storage_) {
        if (allowed == pair) {
            return true;
        }
    }
    return false;
}
