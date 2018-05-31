package de.hpi.machinelearning;

import java.util.*;

public class TextSimilarityCalculator {

    public static double jaccardSimilarity(String left, String right) {
        if (stringsAreEmpty(left, right)) {
            return -1d;
        }

        Set<String> leftWords = new HashSet<>(getNGramm(left, 3));
        Set<String> rightWords = new HashSet<>(getNGramm(right, 3));
        final Set<String> intersectionSet = getIntersection(leftWords, rightWords);
        final Set<String> unionSet = getUnion(leftWords, rightWords);
        return (double)intersectionSet.size() / (double)unionSet.size();
    }

    public static double cosineSimilarity(String left, String right) {
        if (stringsAreEmpty(left, right)) {
            return -1d;
        }

        Map<String, Integer> leftVector = getWordOccurenceVector(getNGramm(left, 3));
        Map<String, Integer> rightVector = getWordOccurenceVector(getNGramm(right, 3));

        final double dotProduct = dot(leftVector, rightVector);
        double d1 = 0.0d;
        for(final Integer value : leftVector.values()) {
            d1 += Math.pow(value, 2);
        }
        double d2 = 0.0d;
        for(final Integer value : rightVector.values()) {
            d2 += Math.pow(value, 2);
        }
        double cosineSimilarity;
        if(d1 <= 0.0 || d2 <= 0.0) {
            cosineSimilarity = 0.0;
        } else {
            cosineSimilarity = dotProduct / (Math.sqrt(d1) * Math.sqrt(d2));
        }
        return cosineSimilarity;
    }

    private static Set<String> getIntersection(final Set<String> leftWords, final Set<String> rightWords) {
        final Set<String> intersection = new HashSet<>(leftWords);
        intersection.retainAll(rightWords);
        return intersection;
    }

    private static Set<String> getUnion(final Set<String> leftWords, final Set<String> rightWords) {
        final Set<String> union = new HashSet<>(leftWords);
        union.addAll(rightWords);
        return union;
    }

    private static double dot(final Map<String, Integer> leftVector, final Map<String, Integer> rightVector) {
        Set<String> intersection = getIntersection(leftVector.keySet(), rightVector.keySet());
        long dotProduct = 0;
        for (final String key : intersection) {
            dotProduct += leftVector.get(key) * rightVector.get(key);
        }
        return dotProduct;
    }

    private static Map<String, Integer> getWordOccurenceVector(List<String> words) {
        Map<String, Integer> resultVector = new HashMap<>();

        for(String word : words) {
            if(resultVector.containsKey(word)) {
                resultVector.put(word, resultVector.get(word) + 1);
            } else {
                resultVector.put(word, 1);
            }
        }
        return resultVector;
    }

    private static List<String> getNGramm(String input, int tokenLength) {
        String padding = String.format("%1$" + (tokenLength - 1) + "s", "");
        String normalisedInput = padding + input.toLowerCase() + padding;
        List<String> result = new ArrayList<>();

        for(int i = 0; i < normalisedInput.length() -  (tokenLength - 1); i++) {
            result.add(normalisedInput.substring(i, i + tokenLength));
        }
        return result;
    }

    private static boolean stringsAreEmpty(String a, String b) {
        return a == null || b == null || stringIsEmpty(a) || stringIsEmpty(b);

    }

    private static boolean stringIsEmpty(String string) {
        String stringWithoutWhites = string.replaceAll("\\w", "");
        return stringWithoutWhites.isEmpty();
    }

}