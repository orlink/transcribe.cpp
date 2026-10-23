// Draft tokens for draft-verified greedy decoding.
//
// A caller may pass text it expects to be close to the transcript, such as a
// first-pass recognizer's output. The decoder feeds the next few draft tokens
// together with its last token in one pass, keeps the longest prefix its own
// argmax agrees with plus the token it predicts after that prefix, and
// continues from there. The output is the greedy output; only the number of
// decoder passes changes.
//
// DraftCursor tracks where the decoder's output stands in the draft. After a
// mismatch it looks for the emitted token in the draft near the place the
// decoder was expected to be, or anywhere when the last few emitted tokens
// match there, and drafts again from the token after it.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace transcribe {

struct DraftRealign {
    // A one-token match counts when it lies this many tokens before or after
    // the expected draft position.
    int window_back = 3;
    int window_fwd  = 4;
    // A match farther away needs this many trailing emitted tokens to match
    // (0: never).
    int far_suffix  = 2;
};

class DraftCursor {
  public:
    void reset(std::vector<int32_t> draft, DraftRealign realign = DraftRealign{}) {
        draft_   = std::move(draft);
        realign_ = realign;
        hist_.clear();
        pos_    = draft_.empty() ? -1 : 0;
        expect_ = 0;
    }

    bool empty() const { return draft_.empty(); }

    // Number of draft tokens (at most k_max) predicted to follow the last
    // emitted token; *out points at the first of them.
    int propose(int k_max, const int32_t ** out) const {
        *out = nullptr;
        if (pos_ < 0 || k_max <= 0) {
            return 0;
        }
        const int left = static_cast<int>(draft_.size()) - pos_;
        const int k    = left < k_max ? left : k_max;
        if (k > 0) {
            *out = draft_.data() + pos_;
        }
        return k > 0 ? k : 0;
    }

    // Report the next token the decoder emitted.
    void push(int32_t tok) {
        hist_.push_back(tok);
        const int n = static_cast<int>(draft_.size());
        if (pos_ >= 0 && pos_ < n && draft_[pos_] == tok) {
            ++pos_;
            expect_ = pos_;
            return;
        }
        const int e    = expect_;
        int       best = -1;
        int       cost = 0;
        for (int i = 0; i < n; ++i) {
            if (draft_[i] != tok) {
                continue;
            }
            const int d         = i - e;
            const int in_window = d >= -realign_.window_back && d <= realign_.window_fwd;
            if (!in_window && (realign_.far_suffix <= 0 || suffix_len(i) < realign_.far_suffix)) {
                continue;
            }
            // Nearest first; on a tie the later position (the decoder skipped
            // draft tokens) wins over the earlier one.
            const int c = d >= 0 ? 2 * d : -2 * d + 1;
            if (best < 0 || c < cost) {
                best = i;
                cost = c;
            }
        }
        if (best >= 0) {
            pos_    = best + 1;
            expect_ = pos_;
        } else {
            pos_    = -1;
            expect_ = e + 1;
        }
    }

  private:
    // Trailing emitted tokens (the last one included) that match the draft
    // ending at index i.
    int suffix_len(int i) const {
        const int h = static_cast<int>(hist_.size());
        int       l = 0;
        while (l < h && i - l >= 0 && draft_[i - l] == hist_[h - 1 - l]) {
            ++l;
        }
        return l;
    }

    std::vector<int32_t> draft_;
    std::vector<int32_t> hist_;
    DraftRealign         realign_;
    int                  pos_    = -1;  // draft index expected next, -1 when lost
    int                  expect_ = 0;   // where the next token would sit in the draft
};

}  // namespace transcribe
