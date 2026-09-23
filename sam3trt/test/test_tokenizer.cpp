#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "sam3trt/Tokenizer.hpp"

static const std::string CONFIG_DIR = SAM3_CONFIG_DIR;

class CLIPTokenizerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tokenizer_ = CLIPTokenizer(CONFIG_DIR + "/merges.txt", CONFIG_DIR + "/vocab.json");
    }

    static CLIPTokenizer tokenizer_;
};

CLIPTokenizer CLIPTokenizerTest::tokenizer_;

TEST_F(CLIPTokenizerTest, ConstructorThrowsOnBadMergesPath) {
    EXPECT_THROW(CLIPTokenizer("/nonexistent/merges.txt", CONFIG_DIR + "/vocab.json"), std::runtime_error);
}

TEST_F(CLIPTokenizerTest, ConstructorThrowsOnBadVocabPath) {
    EXPECT_THROW(CLIPTokenizer(CONFIG_DIR + "/merges.txt", "/nonexistent/vocab.json"), std::runtime_error);
}

TEST_F(CLIPTokenizerTest, LoadsMergesAndVocab) {
    EXPECT_FALSE(tokenizer_.getMerges().empty());
    EXPECT_FALSE(tokenizer_.getVocab().empty());
}

TEST_F(CLIPTokenizerTest, SpecialTokens) {
    EXPECT_EQ(tokenizer_.getSOTToken(), 49406);
    EXPECT_EQ(tokenizer_.getPaddingToken(), 49407);
    EXPECT_EQ(tokenizer_.getVocab().at("<|startoftext|>"), 49406);
    EXPECT_EQ(tokenizer_.getVocab().at("<|endoftext|>"), 49407);
}

TEST_F(CLIPTokenizerTest, EmptyTextIsSotEot) {
    EXPECT_EQ(tokenizer_.tokenize(""), (std::vector<int>{49406, 49407}));
}

TEST_F(CLIPTokenizerTest, SingleWord) {
    EXPECT_EQ(tokenizer_.tokenize("person"), (std::vector<int>{49406, 2533, 49407}));
}

TEST_F(CLIPTokenizerTest, MultipleWords) {
    EXPECT_EQ(tokenizer_.tokenize("red car"), (std::vector<int>{49406, 736, 1615, 49407}));
    EXPECT_EQ(tokenizer_.tokenize("traffic light"), (std::vector<int>{49406, 3399, 1395, 49407}));
}

TEST_F(CLIPTokenizerTest, CaseInsensitive) {
    EXPECT_EQ(tokenizer_.tokenize("Red CAR"), tokenizer_.tokenize("red car"));
}

TEST_F(CLIPTokenizerTest, CollapsesWhitespace) {
    EXPECT_EQ(tokenizer_.tokenize("  red \t car  "), tokenizer_.tokenize("red car"));
}
