package de.hpi.modelgenerator.services;

import de.hpi.machinelearning.TextSimilarityCalculator;
import lombok.AccessLevel;
import lombok.Getter;
import org.junit.Test;

import static org.junit.Assert.*;

public class TextSimilarityCalculatorTest {

    @Getter(AccessLevel.PRIVATE) private static String EXAMPLE_STRING = "abc def ghi jkl";
    @Getter(AccessLevel.PRIVATE) private static String EXAMPLE_STRING2 = "ABC DEF GHI JKL";
    @Getter(AccessLevel.PRIVATE) private static double EXPECTED_SIMILARITY = 1d;

    @Test
    public void jaccardSimilarity() {
        assertEquals(getEXPECTED_SIMILARITY(), TextSimilarityCalculator.jaccardSimilarity(getEXAMPLE_STRING(), getEXAMPLE_STRING2()), 0.01);
    }

    @Test
    public void cosineSimilarity() {
        assertEquals(getEXPECTED_SIMILARITY(), TextSimilarityCalculator.cosineSimilarity(getEXAMPLE_STRING(), getEXAMPLE_STRING2()), 0.01);
    }
}