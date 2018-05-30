package de.hpi.modelgenerator.services;

import de.hpi.machinelearning.TextSimilarityCalculator;
import lombok.AccessLevel;
import lombok.Getter;
import org.junit.Test;

import static org.junit.Assert.*;

public class TextSimilarityCalculatorTest {

    @Getter(AccessLevel.PRIVATE) private static String EXAMPLE_STRING = "abc def ghi jkl";
    @Getter(AccessLevel.PRIVATE) private static String EXAMPLE_DIFFERENT_STRING = "mno pgr stu vwx";
    @Getter(AccessLevel.PRIVATE) private static String EXAMPLE_EMPTY_STRING = "";
    @Getter(AccessLevel.PRIVATE) private static double EXPECTED_MATCH_SIMILARITY = 1d;
    @Getter(AccessLevel.PRIVATE) private static double EXPECTED_DIFFERENCE_SIMILARITY = 0d;
    @Getter(AccessLevel.PRIVATE) private static double EXPECTED_EMPTY_SIMILARITY = -1d;

    @Test
    public void jaccardSimilarityMatch() {
        assertEquals(getEXPECTED_MATCH_SIMILARITY(), TextSimilarityCalculator.jaccardSimilarity(getEXAMPLE_STRING(), getEXAMPLE_STRING().toUpperCase()), 0.01);
    }

    @Test
    public void jaccardSimilarityDifference() {
        assertEquals(getEXPECTED_DIFFERENCE_SIMILARITY(), TextSimilarityCalculator.jaccardSimilarity(getEXAMPLE_STRING(), getEXAMPLE_DIFFERENT_STRING()), 0.01);
    }

    @Test
    public void jaccardSimilarityWithNull() {
        assertEquals(getEXPECTED_EMPTY_SIMILARITY(), TextSimilarityCalculator.jaccardSimilarity(getEXAMPLE_STRING(), null), 0.01);
    }

    @Test
    public void jaccardSimilarityWithEmptyString() {
        assertEquals(getEXPECTED_EMPTY_SIMILARITY(), TextSimilarityCalculator.jaccardSimilarity(getEXAMPLE_STRING(), getEXAMPLE_EMPTY_STRING()), 0.01);
    }

    @Test
    public void cosineSimilarityMatch() {
        assertEquals(getEXPECTED_MATCH_SIMILARITY(), TextSimilarityCalculator.cosineSimilarity(getEXAMPLE_STRING(), getEXAMPLE_STRING().toUpperCase()), 0.01);
    }

    @Test
    public void cosineSimilarityDifference() {
        assertEquals(getEXPECTED_DIFFERENCE_SIMILARITY(), TextSimilarityCalculator.cosineSimilarity(getEXAMPLE_STRING(), getEXAMPLE_DIFFERENT_STRING()), 0.01);
    }

    @Test
    public void cosineSimilarityWithNull() {
        assertEquals(getEXPECTED_EMPTY_SIMILARITY(), TextSimilarityCalculator.cosineSimilarity(getEXAMPLE_STRING(), null), 0.01);
    }

    @Test
    public void cosineSimilarityWithEmptyString() {
        assertEquals(getEXPECTED_EMPTY_SIMILARITY(), TextSimilarityCalculator.cosineSimilarity(getEXAMPLE_STRING(), getEXAMPLE_EMPTY_STRING()), 0.01);
    }

}