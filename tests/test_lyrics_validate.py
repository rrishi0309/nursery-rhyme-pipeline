from nursery.lyrics.validate import check, count_syllables, line_syllables, rhyme_key


class TestCountSyllables:
    def test_single_vowel_group(self):
        assert count_syllables("cat") == 1

    def test_two_vowel_groups(self):
        assert count_syllables("hello") == 2

    def test_silent_trailing_e_is_dropped(self):
        # "like": groups "i", "e" (2) -> trailing e drops one -> 1
        assert count_syllables("like") == 1

    def test_trailing_e_not_dropped_when_only_one_group(self):
        # "the": single group "e" -> stays 1, not 0
        assert count_syllables("the") == 1

    def test_floors_at_one_with_no_vowels(self):
        assert count_syllables("tsk") == 1

    def test_is_case_insensitive(self):
        assert count_syllables("HELLO") == count_syllables("hello")

    def test_adjacent_vowels_count_as_one_group(self):
        # "queue": "ueue" is one contiguous vowel run -> 1 group -> 1 syllable
        assert count_syllables("queue") == 1

    def test_y_counts_as_a_vowel(self):
        assert count_syllables("sky") == 1
        assert count_syllables("happy") == 2


class TestLineSyllables:
    def test_sums_over_words(self):
        assert line_syllables("hello world") == count_syllables("hello") + count_syllables(
            "world"
        )

    def test_ignores_punctuation(self):
        assert line_syllables("Hello, world!") == line_syllables("Hello world")

    def test_empty_line_is_zero(self):
        assert line_syllables("") == 0

    def test_punctuation_only_line_is_zero(self):
        assert line_syllables("...  --- !!!") == 0


class TestRhymeKey:
    def test_star_example_from_spec(self):
        assert rhyme_key("star") == "ar"

    def test_uses_last_word_of_a_line(self):
        assert rhyme_key("the shining star") == rhyme_key("star")

    def test_lowercases(self):
        assert rhyme_key("STAR") == "ar"

    def test_strips_trailing_punctuation(self):
        assert rhyme_key("look at the star!") == "ar"

    def test_word_with_multiple_vowel_groups_uses_last_one(self):
        # "wonder": groups "o", "e" -> from the last group onward -> "er"
        assert rhyme_key("wonder") == "er"

    def test_two_words_that_should_rhyme_share_a_key(self):
        assert rhyme_key("a shining star") == rhyme_key("not too far")

    def test_empty_line_returns_empty_string(self):
        assert rhyme_key("") == ""

    def test_punctuation_only_line_returns_empty_string(self):
        assert rhyme_key("...") == ""


class TestCheck:
    def test_valid_lines_have_no_violations(self):
        # AABB, ~4 syllables/line, tolerance 1
        lines = [
            "the cat sat on the mat",  # ...at
            "he wore a big fun hat",  # ...at
            "the dog ran to the tree",  # ...ee
            "so happy and so free",  # ...ee
        ]
        assert check(lines, target=6, tolerance=2, scheme="AABB") == []

    def test_flags_a_line_that_is_too_short(self):
        lines = ["hi", "star"]
        violations = check(lines, target=6, tolerance=1, scheme="AA")
        assert any("line 1" in v and "syllable" in v for v in violations)

    def test_flags_a_line_that_is_too_long(self):
        lines = ["star", "this line has far too many syllables in it today"]
        violations = check(lines, target=1, tolerance=1, scheme="AA")
        assert any("line 2" in v and "syllable" in v for v in violations)

    def test_within_tolerance_is_not_a_violation(self):
        # "star" is 1 syllable; target 2 +/-1 accepts 1, 2, or 3
        assert check(["star"], target=2, tolerance=1, scheme="A") == []

    def test_flags_broken_rhyme_scheme(self):
        lines = ["a shining star", "a happy day"]  # "ar" vs "ay", scheme AA
        violations = check(lines, target=3, tolerance=3, scheme="AA")
        assert any("rhyme" in v for v in violations)

    def test_matching_rhyme_scheme_has_no_rhyme_violation(self):
        lines = ["a shining star", "not near or far"]  # both end "ar"
        violations = check(lines, target=4, tolerance=4, scheme="AA")
        assert not any("rhyme" in v for v in violations)

    def test_different_scheme_letters_need_not_rhyme(self):
        lines = ["a shining star", "a happy day"]  # "ar" vs "ay", scheme AB
        violations = check(lines, target=3, tolerance=3, scheme="AB")
        assert not any("rhyme" in v for v in violations)

    def test_flags_line_count_mismatch(self):
        violations = check(["only one line"], target=3, tolerance=3, scheme="AABB")
        assert any("expected 4" in v for v in violations)

    def test_empty_lines_and_scheme_is_valid(self):
        assert check([], target=4, tolerance=1, scheme="") == []
