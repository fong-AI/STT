from unittest import TestCase
from streaming import StableText


class StableTextTests(TestCase):
    def test_first_result_is_draft_then_common_prefix_is_confirmed(self):
        state = StableText()
        first = [(0, .4, "xin"), (.4, .8, "chao")]
        confirmed, draft = state.update(first)
        self.assertEqual(confirmed, [])
        self.assertEqual(draft, "xin chao")
        confirmed, draft = state.update(first + [(.8, 1.2, "ban")])
        self.assertEqual([w[2] for w in confirmed], ["xin", "chao"])
        self.assertEqual(draft, "ban")

    def test_changed_draft_is_replaced_and_only_final_text_is_confirmed(self):
        state = StableText()
        state.update([(0, .5, "mot")])
        confirmed, draft = state.update([(0, .5, "hai")])
        self.assertEqual(confirmed, [])
        self.assertEqual(draft, "hai")
        confirmed, draft = state.update([(0, .5, "hai")], final=True)
        self.assertEqual([w[2] for w in confirmed], ["hai"])
        self.assertEqual(draft, "")

    def test_confirmed_words_are_not_repeated_and_real_repetitions_remain(self):
        state = StableText()
        first = [(0, .4, "alo"), (.4, .8, "alo"), (.8, 1.1, "mot")]
        state.update(first)
        confirmed, _ = state.update(first)
        self.assertEqual([w[2] for w in confirmed], ["alo", "alo"])
        confirmed, draft = state.update(first + [(1.1, 1.4, "hai")], final=True)
        self.assertEqual([w[2] for w in confirmed], ["mot", "hai"])
        self.assertEqual(draft, "")

    def test_punctuation_and_case_do_not_prevent_stability(self):
        state = StableText()
        state.update([(0, .4, "Xin,"), (.4, .8, "chao")])
        confirmed, _ = state.update([(0, .4, "xin"), (.4, .8, "chao"), (.8, 1.1, "ban")])
        self.assertEqual([w[2] for w in confirmed], ["xin", "chao"])

    def test_timestamp_drift_does_not_repeat_committed_boundary_word(self):
        state = StableText()
        first = [(0, .4, "xin"), (.4, .8, "chao"), (.8, 1.2, "ban")]
        state.update(first)
        state.update(first)
        confirmed, _ = state.update([(0, .42, "xin"), (.42, .84, "chao"), (.84, 1.2, "ban")], final=True)
        self.assertEqual([w[2] for w in confirmed], ["ban"])

    def test_equal_word_endpoints_do_not_remove_next_word(self):
        state = StableText()
        first = [(0, .4, "xin"), (.4, .4, "chao")]
        state.update(first)
        state.update(first)
        confirmed, _ = state.update(first, final=True)
        self.assertEqual([w[2] for w in confirmed], ["chao"])

    def test_revision_to_earlier_committed_word_does_not_repeat_boundary(self):
        state = StableText()
        first = [(0, .4, "xin"), (.4, .8, "chao"), (.8, 1.2, "ban")]
        state.update(first)
        state.update(first)
        confirmed, _ = state.update([(0, .42, "sin"), (.42, .84, "chao"), (.84, 1.2, "ban")], final=True)
        self.assertEqual([w[2] for w in confirmed], ["ban"])

    def test_new_repetition_after_committed_word_is_preserved_without_old_context(self):
        state = StableText()
        state.update([(0, .2, "alo")], final=True)
        confirmed, _ = state.update([(.2, .4, "alo")], final=True)
        self.assertEqual([w[2] for w in confirmed], ["alo"])
