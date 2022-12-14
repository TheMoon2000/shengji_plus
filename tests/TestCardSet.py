import unittest
import sys
import random

sys.path.append('.')
from env.utils import CardSuit, TrumpSuite
from env.CardSet import CardSet, MoveType

class TestCardSet(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(123)

        self.cardset_simple = CardSet({
            '2♦': 2,
            '2♥': 2,
            '3♦': 2,
            '3♥': 2,
            '4♦': 2,
            '4♥': 2,
            'XJ': 2
        })

        self.cardset_all_suites = CardSet({
            '8♣': 1,
            '10♠': 2,
            'J♠': 2,
            'Q♠': 1,
            '2♦': 2,
            '9♦': 1,
            '10♦': 1,
            '8♥': 1,
            'DJ': 2
        })

        self.cardset_big_suite = CardSet({
            'A♥': 2,
            'K♥': 2,
            'Q♥': 2,
            'J♥': 2,
            '10♥': 2,
            '4♥': 2,
            '2♥': 2,

            '3♣': 2,
        })

        self.cardset_multiple_small_tractors = CardSet({
            'K♦': 2,
            'Q♦': 2,
            '10♦': 2,
            '9♦': 2,
            '7♦': 2,
            '6♦': 2
        })

        return super().setUp()

    def test_total_points(self):
        cardset = CardSet()
        cardset.add_card('10♣')
        cardset.add_card('7♣')
        cardset.add_card('5♠')
        cardset.add_card('5♠')
        self.assertEqual(cardset.total_points(), 20)
    
    def test_shuffle_deck(self):
        cardset, order = CardSet.new_deck()
        cardset2, order2 = cardset.new_deck()
        assert cardset.size == cardset2.size == 108
        assert len(order) == len(cardset)
        assert order != order2
    
    def test_declaration_options(self):
        fullset, _ = CardSet.new_deck()
        self.assertEqual(len(fullset.trump_declaration_options(14)), 6)

        cardset = CardSet({
            '2' + TrumpSuite.CLUB: 1,
            '2' + TrumpSuite.DIAMOND: 2,
            '2' + TrumpSuite.SPADE: 1,
            'DJ': 1,
            'XJ': 1
        })
        self.assertEqual(len(cardset.trump_declaration_options(2)), 3)
    
    def test_lead_actions_jokers(self):
        cardset = CardSet({
            'DJ': 2,
            'XJ': 2,
            '2' + TrumpSuite.DIAMOND: 2,
            '2' + TrumpSuite.CLUB: 2,
            '2' + TrumpSuite.SPADE: 2
        })
        moves1 = cardset.get_leading_moves(TrumpSuite.SPADE, 2)
        moves2 = cardset.get_leading_moves(TrumpSuite.XJ, 2)
        self.assertEqual(len(moves1), 19) # 5 singles, 5 pairs, 4 length-2 tractors, 3 length-3 tractors, 2 length-4 tractors
        self.assertEqual(len(moves2), 17) # 5 singles, 5 pairs, 4 length-2 tractors, 3 length-3 tractor
    
    def test_lead_actions_skip_dominant_rank(self):
        moves = self.cardset_simple.get_leading_moves(dominant_suite=TrumpSuite.HEART, dominant_rank=3)
        self.assertEqual(len(moves), 19) # 7 singles, 7 pairs, 4 length-2 tractors, 1 length-3 tractor

        moves = self.cardset_big_suite.get_leading_moves(dominant_suite=TrumpSuite.HEART, dominant_rank=3)
        self.assertEqual(len(moves), 32)
    
    def test_count_suite(self):
        # for suite in [CardSuite.CLUB, CardSuite.DIAMOND, CardSuite.HEART, CardSuite.SPADE, CardSuite.TRUMP]:
        self.assertEqual(self.cardset_simple.count_suite(CardSuit.TRUMP, TrumpSuite.HEART, 3), 10)
        self.assertEqual(self.cardset_simple.count_suite(CardSuit.HEART, TrumpSuite.HEART, 3), 0) # Since HEART is trump, there is no card in the HEART category, since they would all be trump cards
        self.assertEqual(self.cardset_simple.count_suite(CardSuit.DIAMOND, TrumpSuite.HEART, 3), 4) # 3s are trump cards, not diamond cards
        self.assertEqual(self.cardset_simple.count_suite(CardSuit.HEART, TrumpSuite.DJ, 3), 4)

    def test_matching_moves_single(self):
        # Analysis: 5♥ is a trump card in this situation, so the player's action set contains all single trump cards.
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Single('5♥'), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 5) # XJ, 3♦, 4♥, 3♥, 2♥ are the 5 valid moves
    
        # Analysis: the player has no ♣ suite cards, so they can choose any of the 7 cards to play.
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Single('5♣'), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 7)

        # Analysis: in NT mode, only the 3s and jokers are trump cards. So 3 action choices in total.
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Single('3♠'), TrumpSuite.XJ, 3)
        self.assertEqual(len(matching_moves), 3) # 3♦, 3♥, XJ

        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Single('7♥'), TrumpSuite.XJ, 3)
        self.assertEqual(len(matching_moves), 2) # 4♥ and 2♥
    
    def test_matching_moves_pair(self):
        matching_moves = self.cardset_all_suites.get_matching_moves(MoveType.Pair('2♠'), TrumpSuite.HEART, 2)
        self.assertEqual(len(matching_moves), 2) # The player can play a pair of 2♦ or jokers

        matching_moves = self.cardset_all_suites.get_matching_moves(MoveType.Pair('3♠'), TrumpSuite.HEART, 2)
        self.assertEqual(len(matching_moves), 2) # Can play a pair of 10♠ or J♠

        # Analysis: the player only has 1 CLUB card. They need to play 1 additional card. There are 8 to choose from.
        matching_moves = self.cardset_all_suites.get_matching_moves(MoveType.Pair('3♣'), TrumpSuite.HEART, 2)
        self.assertEqual(len(matching_moves), 8)

        # Analysis: the player has no CLUB cards. So they can pick any two cards to play. If they play a trump pair, it's a RUFF so it's counted as a pair. If they choose any two other cards, it's counted as a passive combo.
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Pair('3♣'), TrumpSuite.HEART, 2)
        self.assertEqual(len(matching_moves), 28) # 7 pairs + 21 two card combos

        matching_moves = self.cardset_all_suites.get_matching_moves(MoveType.Pair('4♦'), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 1)
    
    def test_matching_moves_tractor(self):
        # Analysis: the player has 3 exact matches (tractor of length 2)
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Tractor(CardSet({'5♥': 2, '6♥': 2})), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 3) # Either [2♥ 2♥ 4♥ 4♥], [3♦ 3♦ 3♥ 3♥], or [3♥ 3♥ XJ XJ]

        # Analysis: the player doesn't have a single match in the same suite. So they can choose any 4 cards.
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Tractor(CardSet({'5♠': 2, '6♠': 2})), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 161)

        # Analysis: the player doesn't have a tractor, but has one or more pairs. They have to play them first, then choose cards of the same suite, then any card.
        matching_moves = self.cardset_all_suites.get_matching_moves(MoveType.Tractor(CardSet({'5♦': 2, '6♦': 2})), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 1) # The player has 4 diamonds and needs to play all of them.

        # Analysis: the player has [3♦ 3♦ 3♥ 3♥ XJ XJ] as a length-3 tractor
        matching_moves = self.cardset_simple.get_matching_moves(MoveType.Tractor(CardSet({'5♥': 2, '6♥': 2, '7♥': 2})), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 1)

        matching_moves = self.cardset_big_suite.get_matching_moves(MoveType.Tractor(CardSet({'6♥': 2, '7♥': 2})), TrumpSuite.HEART, 3)
        self.assertEqual(len(matching_moves), 6)

        # Analysis: the player has no length-3 tractors. They need to pick a length-2 tractor, and then pick some other pair.
        matching_moves = self.cardset_multiple_small_tractors.get_matching_moves(MoveType.Tractor(CardSet({'3♦': 2, '4♦': 2, '5♦': 2})), TrumpSuite.HEART, 2)
        self.assertEqual(len(matching_moves), 12)

        # Analysis: the player has no length-4 tractors. They need to pick two length-2 tractors to play.
        matching_moves = self.cardset_multiple_small_tractors.get_matching_moves(MoveType.Tractor(CardSet({'2♦': 2, '3♦': 2, '4♦': 2, '5♦': 2})), TrumpSuite.HEART, 14)
        self.assertEqual(len(matching_moves), 3)


    def test_combo_moves(self):
        decomposition = self.cardset_multiple_small_tractors.decompose(TrumpSuite.DJ, 2)
        self.assertEqual(len(decomposition), 3)

        # If the player can play combos, they can play any card combination of the same suite.
        small_cardset = CardSet({'A♥': 1, 'K♥': 2, 'Q♥': 2, 'A♠': 1})
        combo_moves = small_cardset.get_leading_moves(TrumpSuite.DJ, 2, include_combos=True)
        self.assertEqual(len(combo_moves), 18) # 4 singles + 2 pairs + 1 tractor + 3 length-2 combos + 5 length-3 combos + 2 length-4 combos + 1 length-5 combo 

        combo_moves = self.cardset_multiple_small_tractors.get_leading_moves(TrumpSuite.DJ, 2, include_combos=True)
        self.assertEqual(len(combo_moves), 728) # Hard to tell if this is right

        combo_lead_move = MoveType.Combo(CardSet({'A♦': 2, 'K♦': 1}))
        matching_moves = self.cardset_all_suites.get_matching_moves(combo_lead_move, TrumpSuite.DJ, 3)
        self.assertEqual(len(matching_moves), 2) # Must play the pair of 2s, then choose either 9♦ or 10♦

    def test_cardset_compare_single(self):
        # Single card can beat another single card if the rank is higher, or if it's a trump card but the other one is not
        single_k_move = MoveType.Single('K♦')
        single_ace = CardSet({'A♦': 1})
        single_queen = CardSet({'Q♦': 1})
        single_jack_heart = CardSet({'J♥': 1})
        single_k_heart = CardSet({'K♥': 1})
        
        # Same suite cards. Compare rank only.
        self.assertTrue(all([single_ace.is_bigger_than(single_k_move, suite, 2) for suite in (TrumpSuite.CLUB, TrumpSuite.DIAMOND, TrumpSuite.DJ, TrumpSuite.HEART, TrumpSuite.SPADE)]))
        self.assertFalse(all([single_queen.is_bigger_than(single_k_move, suite, 2) for suite in (TrumpSuite.CLUB, TrumpSuite.DIAMOND, TrumpSuite.DJ, TrumpSuite.HEART, TrumpSuite.SPADE)]))

        # Trump against non-trump (ruff)
        self.assertTrue(single_jack_heart.is_bigger_than(single_k_move, TrumpSuite.HEART, 2))

        # Non-trump against non-trump of different suite
        self.assertFalse(single_jack_heart.is_bigger_than(single_k_move, TrumpSuite.CLUB, 2))

        # Dominant rank is larger than other non-joker trump cards
        self.assertTrue(single_queen.is_bigger_than(single_k_move, TrumpSuite.DIAMOND, 12))

        # Dominant card should be larger than other dominant-rank cards
        self.assertTrue(single_k_heart.is_bigger_than(single_k_move, TrumpSuite.HEART, 13))
        self.assertFalse(single_k_heart.is_bigger_than(single_k_move, TrumpSuite.XJ, 13))
        self.assertFalse(single_k_heart.is_bigger_than(single_k_move, TrumpSuite.XJ, 12))
    
    def test_cardset_compare_pair(self):
        pair_ace_diamond = CardSet({'A♦': 2})
        pair_king_heart = CardSet({'K♥': 2})
        pair_queen_diamond = CardSet({'Q♦': 2})
        pair_jokers = CardSet({'XJ': 2})
        non_pair_jokers = CardSet({'XJ': 1, 'DJ': 1})

        target_move = MoveType.Pair('K♦')

        self.assertTrue(all([pair_ace_diamond.is_bigger_than(target_move, suite, 2) for suite in (TrumpSuite.CLUB, TrumpSuite.DIAMOND, TrumpSuite.DJ, TrumpSuite.HEART, TrumpSuite.SPADE)]))
        self.assertFalse(all([pair_king_heart.is_bigger_than(target_move, suite, 2) for suite in (TrumpSuite.CLUB, TrumpSuite.DIAMOND, TrumpSuite.DJ, TrumpSuite.SPADE)]))
        self.assertTrue(pair_king_heart.is_bigger_than(target_move, TrumpSuite.HEART, 2))
        self.assertTrue(pair_king_heart.is_bigger_than(target_move, TrumpSuite.HEART, 13))
        self.assertFalse(pair_queen_diamond.is_bigger_than(target_move, TrumpSuite.SPADE, 2))

        # Two different jokers cannot beat any pair
        self.assertFalse(non_pair_jokers.is_bigger_than(target_move, TrumpSuite.DJ, 2))

        # Pair jokers beat any non-joker pair in any suite and any rank
        self.assertTrue(all([pair_jokers.is_bigger_than(target_move, suite, 13) for suite in (TrumpSuite.CLUB, TrumpSuite.DIAMOND, TrumpSuite.DJ, TrumpSuite.HEART, TrumpSuite.SPADE)]))

    def test_cardset_compare_tractor(self):
        target_tractor = CardSet({ '6♦': 2, '8♦': 2 }) # Assume 7 is dominant rank
        target_move = MoveType.Combo(target_tractor) # Could use MoveType.Tractor(...) also

        option1 = CardSet({ '7♥': 2, '7♦': 2 })
        option2 = CardSet({ 'XJ': 2, '7♥': 2 })
        option3 = CardSet({ '10♦': 2, 'J♦': 2 })
        
        # If either ♥ or ♦ is trump suite, then option1 forms a tractor
        self.assertTrue(all([option1.is_bigger_than(target_move, suite, 7) for suite in (TrumpSuite.DIAMOND, TrumpSuite.HEART)]))

        # Otherwise, 7♦7♦ 7♥7♥ is not a tractor
        self.assertFalse(all([option1.is_bigger_than(target_move, suite, 7) for suite in (TrumpSuite.DJ, TrumpSuite.SPADE, TrumpSuite.CLUB)]))

        # Option2 is a tractor if the trump suite is ♥ or NT 
        self.assertTrue(all([option2.is_bigger_than(target_move, suite, 7) for suite in (TrumpSuite.HEART, TrumpSuite.XJ)]))
        self.assertFalse(all([option2.is_bigger_than(target_move, suite, 7) for suite in (TrumpSuite.CLUB, TrumpSuite.SPADE, TrumpSuite.DIAMOND)]))

        # Option3 is always a tractor given dominant rank = 7, and beats the target tractor
        self.assertTrue(all([option3.is_bigger_than(target_move, suite, 7) for suite in (TrumpSuite.CLUB, TrumpSuite.DIAMOND, TrumpSuite.HEART, TrumpSuite.SPADE, TrumpSuite.DJ)]))

    def test_cardset_compare_combo(self):
        target1 = CardSet({'A♦': 1, 'K♦': 2})
        target2 = CardSet({'A♦': 1, 'K♦': 1, '8♦': 2, '9♦': 2})
        
        option1 = CardSet({'3♥': 2, '4♥': 1})
        option2 = CardSet({'3♥': 2, '5♥': 2, '7♥': 2})

        # Since 3♦ is a dominant card pair, it's larger than a pair of K♦. The single card doesn't matter in this case.
        self.assertTrue(option1.is_bigger_than(MoveType.Combo(target1), TrumpSuite.HEART, 3))
        self.assertTrue(option1.is_bigger_than(MoveType.Combo(target1), TrumpSuite.HEART, 4))
        self.assertFalse(option1.is_bigger_than(MoveType.Combo(target1), TrumpSuite.DIAMOND, 3))
        self.assertFalse(option1.is_bigger_than(MoveType.Combo(target1), TrumpSuite.DJ, 2))

        # The player beats the target if they have a tractor and all cards are trump cards.
        self.assertTrue(option2.is_bigger_than(MoveType.Combo(target2), TrumpSuite.HEART, 6))
        self.assertFalse(option2.is_bigger_than(MoveType.Combo(target2), TrumpSuite.HEART, 2))
        self.assertFalse(option2.is_bigger_than(MoveType.Combo(target2), TrumpSuite.DJ, 6))

    def test_multi_compare_combo(self):
        # Player0 hands leading position to player2 in an NT game
        player0_cards = CardSet({'10♥': 1})
        player1_cards = CardSet({'A♥': 1}) # beats player0
        player2_cards = CardSet({'2♦': 1}) # Ruff
        player3_cards = CardSet({'4♥': 1}) # follows
        self.assertEqual(CardSet.round_winner([player0_cards, player1_cards, player2_cards, player3_cards], TrumpSuite.XJ, 2), 2)

        # The player with the single largest card wins
        player0_cards = CardSet({ 'A♥': 1, 'K♥': 1 }) # Combo
        player1_cards = CardSet({ '2♠': 1, 'A♠': 1 }) # Ruff
        player2_cards = CardSet({ 'XJ': 1, '5♠': 1 }) # Bigger ruff
        player3_cards = CardSet({ 'Q♥': 1, '6♦': 1 }) # nothing

        # Player 2 has the largest ruff, therefore wins the trick.
        self.assertEqual(CardSet.round_winner([player0_cards, player1_cards, player2_cards, player3_cards], TrumpSuite.SPADE, 2), 2)

        # Player with the largest trump pair wins (given that they play 5 trump cards containing 2 pairs)
        player0_cards = CardSet({ 'A♥': 2, 'K♥': 1, 'Q♥': 2 }) # Combo with 2 pairs and 1 single
        player1_cards = CardSet({ '5♥': 2, '3♥': 1, '7♥': 1, 'J♥': 1 }) # Match suite
        player2_cards = CardSet({ '6♠': 2, '7♠': 2, '2♣': 1 }) # Uses a tractor to ruff the combo
        player3_cards = CardSet({ 'A♠': 2, '5♠': 2, '10♠': 1 }) # Beats player2 because only the largest pair matters here.
        self.assertEqual(CardSet.round_winner([player0_cards, player1_cards, player2_cards, player3_cards], TrumpSuite.SPADE, 2), 3)

        # Player with the largest tractor wins
        player0_cards = CardSet({ '9♥': 2, '10♥': 2 }) # 1 tractor
        player1_cards = CardSet({ 'A♥': 2, 'K♥': 2 }) # Beats player0 in same suite
        player2_cards = CardSet({ '6♠': 2, '7♠': 2 }) # Ruffs using a tractor
        player3_cards = CardSet({ '2♠': 2, '2♣': 2 }) # Ruffs using bigger tractor
        self.assertEqual(CardSet.round_winner([player0_cards, player1_cards, player2_cards, player3_cards], TrumpSuite.SPADE, 2), 3)

        # Example where player0 is the biggest
        player0_cards = CardSet({'Q♥': 2})
        player1_cards = CardSet({'3♥': 2})
        player2_cards = CardSet({ '10♠': 1, 'J♥': 1 }) # player2 escapes 10 points in trump suite
        player3_cards = CardSet({ '5♥': 1, '9♥': 1 })
        self.assertEqual(CardSet.round_winner([player0_cards, player1_cards, player2_cards, player3_cards], TrumpSuite.SPADE, 2), 0)

    def test_tensor_conversion(self):
        t1 = self.cardset_simple.tensor
        self.assertEqual(self.cardset_simple, CardSet.from_tensor(t1))

        t2 = self.cardset_all_suites.tensor
        self.assertEqual(self.cardset_all_suites, CardSet.from_tensor(t2))
    
    def test_multiplier(self):
        base1 = MoveType.Combo(CardSet({ 'DJ': 1 }))
        base2 = MoveType.Combo(CardSet({ 'A♥': 1, 'K♥': 1 }))
        self.assertEqual(base1.get_multiplier(TrumpSuite.DJ, 2), 2)
        self.assertEqual(base2.get_multiplier(TrumpSuite.CLUB, 2), 2)

        double1 = MoveType.Combo(CardSet({ 'A♥': 2, 'K♥': 1 }))
        double2 = MoveType.Combo(CardSet({ '2♥': 2, '2♣': 2 }))
        self.assertEqual(double1.get_multiplier(TrumpSuite.CLUB, 2), 4)
        self.assertEqual(double2.get_multiplier(TrumpSuite.DJ, 2), 4) # Not a tractor
        self.assertEqual(double2.get_multiplier(TrumpSuite.CLUB, 2), 16) # Tractor

        tractor1 = MoveType.Combo(CardSet({ 'A♥': 1, '8♥': 2, '9♥': 2, '10♥': 2 }))
        self.assertEqual(tractor1.get_multiplier(TrumpSuite.HEART, 2), 64)

    def test_legal_combo(self):
        leading_move = MoveType.Combo(CardSet({'A♣': 1, 'Q♣': 2}))
        player1 = CardSet({'K♣': 2, '10♣': 1, '4♣': 1})
        player2 = CardSet({'J♣': 2, 'DJ': 1 })
        player3 = CardSet({'A♣': 1, '9♣': 2})

        self.assertEqual(CardSet.is_legal_combo(leading_move, [player1, player2, player3], TrumpSuite.HEART, 2), (False, CardSet({'Q♣': 2})))
        self.assertEqual(CardSet.is_legal_combo(leading_move, [player2, player3], TrumpSuite.DIAMOND, 2), (True, None))
        self.assertEqual(CardSet.is_legal_combo(leading_move, [player2], TrumpSuite.CLUB, 2), (False, CardSet({'A♣': 1})))
        self.assertEqual(CardSet.is_legal_combo(leading_move, [player1, player2], TrumpSuite.CLUB, 4), (False, CardSet({'A♣': 1})))

        leading_move = MoveType.Combo(CardSet({'K♣': 1, '6♣': 2, '7♣': 2}))
        player1 = CardSet({'A♣': 1})
        player2 = CardSet({'8♣': 2, '10♣': 2, 'K♣': 1})
        player3 = CardSet({'DJ': 2, 'XJ': 2, '2♣': 1})

        self.assertEqual(CardSet.is_legal_combo(leading_move, [player1], TrumpSuite.DIAMOND, 2), (False, CardSet({'K♣': 1})))
        self.assertEqual(CardSet.is_legal_combo(leading_move, [player2, player3], TrumpSuite.HEART, 2), (True, None))
        self.assertEqual(CardSet.is_legal_combo(leading_move, [player2], TrumpSuite.HEART, 9), (False, CardSet({'6♣': 2, '7♣': 2})))
        self.assertEqual(CardSet.is_legal_combo(leading_move, [player3], TrumpSuite.CLUB, 2), (False, CardSet({'K♣': 1})))

    def test_trump_suite_tensor(self):
        self.assertEqual(TrumpSuite.from_tensor(TrumpSuite.DIAMOND.tensor), TrumpSuite.DIAMOND)
        self.assertEqual(TrumpSuite.from_tensor(TrumpSuite.CLUB.tensor), TrumpSuite.CLUB)
        self.assertEqual(TrumpSuite.from_tensor(TrumpSuite.HEART.tensor), TrumpSuite.HEART)
        self.assertEqual(TrumpSuite.from_tensor(TrumpSuite.SPADE.tensor), TrumpSuite.SPADE)
        self.assertEqual(TrumpSuite.from_tensor(TrumpSuite.XJ.tensor), TrumpSuite.XJ)
        self.assertEqual(TrumpSuite.from_tensor(TrumpSuite.DJ.tensor), TrumpSuite.DJ)
    
    def test_dynamic_tensor(self):
        full_deck, order = CardSet.new_deck()

        for suit in [TrumpSuite.DIAMOND, TrumpSuite.CLUB, TrumpSuite.XJ, TrumpSuite.DJ]:
            for rank in range(2, 15):
                self.assertEqual(CardSet.from_dynamic_tensor(full_deck.get_dynamic_tensor(suit, rank), suit, rank), full_deck)
        
        half_deck = CardSet()
        for card in order[:28]:
            half_deck.add_card(card)
        
        for suit in [TrumpSuite.HEART, TrumpSuite.SPADE, TrumpSuite.XJ, TrumpSuite.DJ]:
            for rank in range(2, 15):
                self.assertEqual(CardSet.from_dynamic_tensor(half_deck.get_dynamic_tensor(suit, rank), suit, rank), half_deck)
    
    def test_random_hand(self):
        random.seed(10)
        full_deck, order = CardSet.new_deck()
        cardset = CardSet()
        for card in order[:25]:
            cardset.add_card(card)
        print(cardset)

if __name__ == '__main__':
    unittest.main()