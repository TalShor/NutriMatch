"""Exact food description matching with minimal string manipulations.

This module provides fast exact matching for food descriptions by normalizing
them to canonical forms and performing string equality checks.
"""

import re
from dataclasses import dataclass
from typing import Dict, List

import inflect


@dataclass
class MatchResult:
    """Result of exact matching operation."""

    is_exact_match: bool
    normalized_reference: str
    normalized_candidate: str
    confidence: float = 1.0  # Always 1.0 for exact matches


class FoodDescriptionNormalizer:
    """Normalizes food descriptions to canonical forms for exact matching."""

    def __init__(self):
        # Initialize inflect engine for plural/singular conversion
        self.inflect_engine = inflect.engine()

        # Define normalization mappings - order matters for multi-word phrases
        self.synonyms = {
            # Multi-word salt phrases (must come before single words)
            "with salt added": "salted",
            "without salt added": "unsalted",
            "with salt": "salted",
            "without salt": "unsalted",
            "no salt added": "unsalted",
            "low sodium": "low salt",
            # Multi-word cooking phrases
            "pan fried": "fried",
            "deep fried": "fried",
            "stir fried": "fried",
            "fat not added": "no fat",
            "without added fat": "no fat",
            "with fat": "fat added",
            "cooked with oil": "fat added",
            # Multi-word processing phrases
            "freeze dried": "dehydrated",
            "with added sugar": "sweetened",
            "without added sugar": "unsweetened",
            "sugar free": "unsweetened",
            "no sugar added": "unsweetened",
            "variety meats and by products": "",  # Remove this verbose phrase
            # Spelling variants
            "portabella": "portabello",  # Standardize spelling
            # Bread variants
            "white wheat": "white",  # Simplify bread type
            # Single word cooking methods
            "boiled": "cooked",
            "steamed": "cooked",
            "simmered": "cooked",
            "roasted": "baked",
            "grilled": "broiled",
            # Preparation states
            "uncooked": "raw",
            "unprepared": "raw",
            # Single word processing
            "canned": "preserved",
            "jarred": "preserved",
            "dried": "dehydrated",
            "sweetened": "with sugar",
            "unsweetened": "no sugar",
        }

        # Words to remove (stop words for food descriptions)
        self.stop_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "by",
            "from",
            "includes",
            "contains",
            "may contain",
            "prepared",
            "ready",
            "commercial",
            "brand",
            "style",
            "type",
            "variety",
            "mixed",
            "all",
            "whole",
            "part",
            "piece",
            "pieces",
            "sliced",
            "chopped",
            "diced",
            "minced",
            "ground",
            "crushed",
            "mashed",
            "pureed",
            "usda",
            "foods",
            "food",
            "distribution",
            "program",
            "products",
            # Category prefixes to remove
            "fish",
            "spices",
            # Descriptive words that add no nutritional value
            "with",
            "freshwater",
            "mixed",
        }

        # Category words to remove when specific items are present
        self.conditional_stop_words = {
            "nut": ["almond", "walnut", "pecan", "cashew", "hazelnut", "pistachio"],
            "berry": [
                "blueberry",
                "strawberry",
                "blackberry",
                "raspberry",
                "cranberry",
            ],
            "bean": [
                "kidney",
                "black",
                "pinto",
                "navy",
                "lima",
                "garbanzo",
                "chickpea",
            ],
            "ground": [  # Remove "ground" when referring to spices
                "turmeric",
                "ginger",
                "nutmeg",
                "cinnamon",
                "paprika",
                "cumin",
                "coriander",
                "cardamom",
                "cloves",
                "allspice",
                "pepper",
                "chili",
            ],
            "spice": [  # Remove "spice" when specific spice is mentioned
                "turmeric",
                "ginger",
                "nutmeg",
                "cinnamon",
                "paprika",
                "cumin",
                "coriander",
                "cardamom",
                "cloves",
                "allspice",
                "pepper",
                "chili",
            ],
        }

        # Standardize measurement units
        self.unit_mappings = {
            "oz": "ounce",
            "lb": "pound",
            "lbs": "pound",
            "g": "gram",
            "kg": "kilogram",
            "ml": "milliliter",
            "l": "liter",
            "fl oz": "fluid ounce",
            "tsp": "teaspoon",
            "tbsp": "tablespoon",
            "cup": "cup",
            "cups": "cup",
        }

    def _to_singular(self, word: str) -> str:
        """Convert word to singular form using inflect library."""
        try:
            # Skip if already in manual mappings (higher priority)
            if word in self.synonyms:
                return self.synonyms[word]

            # Use inflect to convert to singular
            singular = self.inflect_engine.singular_noun(word)
            return (
                singular if singular else word
            )  # singular is False if already singular
        except Exception:
            return word  # Fallback to original on any error

    def normalize(self, description: str) -> str:
        """Normalize a food description to canonical form."""
        if not description:
            return ""

        # Convert to lowercase
        text = description.lower().strip()

        # Remove parenthetical information that's often inconsistent
        # But first extract content from parentheses and add it back (for cases like "(raw)")
        parenthetical_content = re.findall(r"\(([^)]*)\)", text)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\[[^\]]*\]", "", text)

        # Add back simple parenthetical content (like raw, cooked, etc.)
        for content in parenthetical_content:
            content = content.strip()
            if content and len(content.split()) <= 2:  # Only short descriptive terms
                text += " " + content

        # Remove extra whitespace and special characters including commas
        text = re.sub(r"[^\w\s-]", " ", text)  # Remove commas and other punctuation
        text = re.sub(r"\s+", " ", text).strip()

        # Apply multi-word phrase replacements first (order matters)
        for phrase, replacement in self.synonyms.items():
            if " " in phrase:  # Multi-word phrase
                text = text.replace(phrase, replacement)

        # Split into tokens for further processing
        tokens = text.split()

        # Process tokens: synonyms -> units -> plurals, filter stop words
        normalized_tokens = []
        for token in tokens:
            if token in self.stop_words:
                continue  # Skip stop words
            elif token in self.synonyms and " " not in self.synonyms[token]:
                # Apply single-word synonyms (highest priority)
                replacement = self.synonyms[token]
                if replacement not in normalized_tokens:
                    normalized_tokens.append(replacement)
            elif token in self.unit_mappings:
                # Apply unit mappings
                mapped = self.unit_mappings[token]
                if mapped not in normalized_tokens:
                    normalized_tokens.append(mapped)
            else:
                # Apply automatic plural→singular conversion (lowest priority)
                singular_token = self._to_singular(token)
                if singular_token not in normalized_tokens:
                    normalized_tokens.append(singular_token)

        # Apply conditional stop word removal
        # Remove category words when specific items are present
        tokens_to_remove = set()
        for category_word, specific_items in self.conditional_stop_words.items():
            if category_word in normalized_tokens:
                # Check if any specific item is present
                for specific_item in specific_items:
                    if specific_item in normalized_tokens:
                        tokens_to_remove.add(category_word)
                        break

        # Remove the category words
        normalized_tokens = [
            token for token in normalized_tokens if token not in tokens_to_remove
        ]

        # Sort tokens for consistent ordering (key insight for exact matching)
        result = sorted(normalized_tokens) if normalized_tokens else []

        return " ".join(result)


class ExactFoodMatcher:
    """Fast exact matching for food descriptions."""

    def __init__(self):
        self.normalizer = FoodDescriptionNormalizer()

    def find_exact_matches(self, reference: str, candidates: List[str]) -> List[bool]:
        """Find exact matches between reference and candidates.

        Args:
            reference: Reference food description
            candidates: List of candidate food descriptions

        Returns:
            List of boolean values indicating exact matches
        """
        normalized_ref = self.normalizer.normalize(reference)
        results = []

        for candidate in candidates:
            normalized_candidate = self.normalizer.normalize(candidate)
            is_match = normalized_ref == normalized_candidate
            results.append(is_match)

        return results

    def get_match_details(
        self, reference: str, candidates: List[str]
    ) -> List[MatchResult]:
        """Get detailed match results including normalized forms.

        Args:
            reference: Reference food description
            candidates: List of candidate food descriptions

        Returns:
            List of MatchResult objects with details
        """
        normalized_ref = self.normalizer.normalize(reference)
        results = []

        for candidate in candidates:
            normalized_candidate = self.normalizer.normalize(candidate)
            is_match = normalized_ref == normalized_candidate

            result = MatchResult(
                is_exact_match=is_match,
                normalized_reference=normalized_ref,
                normalized_candidate=normalized_candidate,
                confidence=1.0 if is_match else 0.0,
            )
            results.append(result)

        return results

    def precompute_normalized_forms(self, descriptions: List[str]) -> Dict[str, str]:
        """Precompute normalized forms for batch processing.

        Args:
            descriptions: List of food descriptions

        Returns:
            Dictionary mapping original -> normalized description
        """
        return {desc: self.normalizer.normalize(desc) for desc in descriptions}


def test_exact_matcher():
    """Test the exact matching functionality."""
    matcher = ExactFoodMatcher()

    # Show normalization examples
    print("=== NORMALIZATION EXAMPLES ===")
    test_phrases = [
        "almonds, dry roasted, salted",
        "nuts, almonds, dry roasted, with salt added",
        "bananas, raw",
        "banana, raw",
        "beef, variety meats and by products, liver, raw",
        "beef liver, raw",
    ]

    for phrase in test_phrases:
        normalized = matcher.normalizer.normalize(phrase)
        print(f"'{phrase}' -> '{normalized}'")
    print()

    # Test cases
    test_cases = [
        {
            "reference": "almonds, dry roasted, salted",
            "candidates": [
                "nuts, almonds, dry roasted, with salt added",  # Should match
                "nuts, almonds, oil roasted, with salt added",  # Should not match
                "almonds, dry roasted, without salt added",  # Should not match
            ],
            "expected": [True, False, False],
        },
        {
            "reference": "bananas, raw",
            "candidates": [
                "banana, raw",  # Should match (singular/plural)
                "plantains, raw",  # Should not match
                "bananas, cooked",  # Should not match
            ],
            "expected": [True, False, False],
        },
        {
            "reference": "beef, variety meats and by products, liver, raw",
            "candidates": [
                "beef liver, raw",  # Should match
                "beef liver, cooked",  # Should not match
                "chicken liver, raw",  # Should not match
            ],
            "expected": [True, False, False],
        },
    ]

    print("Testing Exact Food Matcher")
    print("=" * 50)

    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['reference']}")
        print("-" * 40)

        results = matcher.find_exact_matches(test["reference"], test["candidates"])
        details = matcher.get_match_details(test["reference"], test["candidates"])

        for j, (candidate, result, expected, detail) in enumerate(
            zip(test["candidates"], results, test["expected"], details)
        ):
            status = "✅" if result == expected else "❌"
            print(f"{status} {candidate}")
            print(f"   Result: {result}, Expected: {expected}")
            print(f"   Normalized ref: '{detail.normalized_reference}'")
            print(f"   Normalized can: '{detail.normalized_candidate}'")
            print()


if __name__ == "__main__":
    test_exact_matcher()
