// Draft cursor (pure host, no model): what a drafted decode proposes after
// the decoder agrees with the draft, inserts, drops or replaces tokens. See
// src/transcribe-draft.h.

#include "transcribe-draft.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void expect(const char * what, bool ok) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

// The draft tokens the cursor proposes next, at most k.
std::vector<int32_t> proposal(const transcribe::DraftCursor & c, int k) {
    const int32_t * p = nullptr;
    const int       n = c.propose(k, &p);
    return std::vector<int32_t>(p, p + n);
}

void push_all(transcribe::DraftCursor & c, const std::vector<int32_t> & toks) {
    for (int32_t t : toks) {
        c.push(t);
    }
}

void test_agreement() {
    transcribe::DraftCursor c;
    c.reset({ 10, 11, 12, 13, 2 });
    expect("starts at the first draft token", proposal(c, 3) == std::vector<int32_t>({ 10, 11, 12 }));
    push_all(c, { 10, 11 });
    expect("advances with agreeing tokens", proposal(c, 8) == std::vector<int32_t>({ 12, 13, 2 }));
    push_all(c, { 12, 13, 2 });
    expect("nothing left at the end", proposal(c, 8).empty());
}

void test_inserted_token() {
    // The decoder adds a token (punctuation) the draft lacks.
    transcribe::DraftCursor c;
    c.reset({ 10, 11, 12, 13, 2 });
    push_all(c, { 10, 99 });
    expect("an unknown token stops drafting", proposal(c, 2).empty());
    push_all(c, { 11 });
    expect("rejoins after an insertion", proposal(c, 2) == std::vector<int32_t>({ 12, 13 }));
}

void test_dropped_token() {
    // The decoder skips a draft token (a filler word).
    transcribe::DraftCursor c;
    c.reset({ 10, 50, 11, 12, 2 });
    push_all(c, { 10, 11 });
    expect("skips a dropped draft token", proposal(c, 2) == std::vector<int32_t>({ 12, 2 }));
}

void test_replaced_token() {
    // The decoder writes a different word.
    transcribe::DraftCursor c;
    c.reset({ 10, 11, 12, 13, 14, 2 });
    push_all(c, { 10, 98, 12 });
    expect("rejoins after a replacement", proposal(c, 2) == std::vector<int32_t>({ 13, 14 }));
}

void test_far_match() {
    transcribe::DraftRealign r;
    r.window_back = 1;
    r.window_fwd  = 1;
    r.far_suffix  = 2;
    transcribe::DraftCursor c;
    c.reset({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 2 }, r);
    push_all(c, { 1, 20 });
    expect("a far single-token match does not count", proposal(c, 1) != std::vector<int32_t>({ 21 }));
    push_all(c, { 21 });
    expect("a far match with enough context counts", proposal(c, 1) == std::vector<int32_t>({ 22 }));
}

void test_nearest_match() {
    transcribe::DraftCursor c;
    c.reset({ 5, 6, 7, 8, 9, 7, 1, 2 });
    push_all(c, { 5, 6, 3 });
    // Token 7 was expected at index 3; it sits at 2 and 5.
    push_all(c, { 7 });
    expect("the nearest occurrence wins", proposal(c, 1) == std::vector<int32_t>({ 8 }));
}

void test_empty_draft() {
    transcribe::DraftCursor c;
    c.reset({});
    expect("empty draft proposes nothing", proposal(c, 4).empty());
    push_all(c, { 1, 2, 3 });
    expect("empty draft stays empty", proposal(c, 4).empty());
}

}  // namespace

int main() {
    test_agreement();
    test_inserted_token();
    test_dropped_token();
    test_replaced_token();
    test_far_match();
    test_nearest_match();
    test_empty_draft();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("draft cursor: ok\n");
    return 0;
}
