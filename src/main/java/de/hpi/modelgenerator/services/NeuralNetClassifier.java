package de.hpi.modelgenerator.services;



import de.hpi.machinelearning.LabelSeeker;
import de.hpi.machinelearning.MeansBuilder;
import de.hpi.modelgenerator.properties.ModelGeneratorProperties;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.SimpleLabelAwareIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.springframework.stereotype.Service;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
@Slf4j
@Service
@RequiredArgsConstructor

class NeuralNetClassifier {

    private final ModelGeneratorProperties properties;

    ParagraphVectors getParagraphVectors(List<LabelledDocument> documents) {

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        ParagraphVectors paragraphVectors = (new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(50)
                .iterate(new SimpleLabelAwareIterator(documents))
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build());

        paragraphVectors.fit();
        return paragraphVectors;
    }

    void checkUnlabeledData(ParagraphVectors paragraphVectors, List<LabelledDocument> testingSet, String modelName) {
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(paragraphVectors.getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());
        int rightMatches = 0;
        int wrongMatches = 0;
        int notLabeled = 0;
        double labelThreshold;
        if (modelName == "brand") {
            labelThreshold = getProperties().getLabelThresholdBrand();
        } else {
            labelThreshold = getProperties().getLabelThresholdCategory();
        }


        List<String> labels = paragraphVectors.getLabelsSource().getLabels();

        Integer numLabels = labels.size();

        int[][] confMatrix = new int[numLabels][numLabels];
        int[] notPredicted = new int[numLabels];

        Set<String> usedLabels = new HashSet();
        for(LabelledDocument document : testingSet) {
            if (!labels.contains(document.getLabels().get(0))) {
                continue;
            }
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);
            
            Pair<String, Double> bestLabel = getBestScoredLabel(scores);
            if(bestLabel != null) {
                if (bestLabel.getRight() < labelThreshold) {
                    notLabeled++;
                    notPredicted[labels.indexOf(document.getLabels().get(0))] += 1;
                } else if (bestLabel.getLeft().equals(document.getLabels().get(0))) {
                    rightMatches++;
                    addToConfMatrix(confMatrix, document.getLabels().get(0), document.getLabels().get(0), labels);
                } else {
                    wrongMatches++;
                    addToConfMatrix(confMatrix, document.getLabels().get(0), bestLabel.getLeft(), labels);
                }
            } else {
                notLabeled++;
                notPredicted[labels.indexOf(document.getLabels().get(0))] += 1;
            }
            usedLabels.add(document.getLabels().get(0));

        }
        
        double weightedPrecision = calcWeightedPrecision(confMatrix, notPredicted, usedLabels);
        double weightedRecall = calcWeightedRecall(confMatrix, notPredicted, usedLabels);
        double weightedAccuracy = calcWeightedAccuracy(confMatrix, notPredicted, usedLabels);
        log.info("Classification Error: {}", (double) wrongMatches / (double) (wrongMatches + rightMatches));
        log.info("Precision: {}", weightedPrecision);
        log.info("Not labeled: {}", notLabeled);
        log.info("Recall: {}", weightedRecall);
        log.info("Different labels: {}", usedLabels.size());
        log.info("F1 Measure: {}", (2* (weightedRecall*weightedPrecision)/(weightedRecall+weightedPrecision)));
        log.info("Accuracy: {}" , weightedAccuracy);

    }

    private double calcWeightedRecall(int[][] confMatrix, int[] notPredicted, Set<String> usedLabels) {
        double[] recalls = new double[notPredicted.length];
        for (int i = 0; i < notPredicted.length; i++) {
            recalls[i] = (double) confMatrix[i][i] / ((double) IntStream.of(confMatrix[i]).sum() );
            if (Double.isNaN(recalls[i])) {
                recalls[i] = 0;
            }
        }
        return DoubleStream.of(recalls).sum() / (double) usedLabels.size();
    }

    private double calcWeightedPrecision(int[][] confMatrix, int[] notPredicted, Set<String> usedLabels) {
        double[] precision = new double[notPredicted.length];
        for (int i = 0; i < notPredicted.length; i++) {
            int totalPredictedLabel = 0;
            for (int j = 0; j < notPredicted.length; j++) {
                    totalPredictedLabel += confMatrix[j][i];
            }
            precision[i] = (double) confMatrix[i][i] / (double) totalPredictedLabel;
            if (Double.isNaN(precision[i])) {
                precision[i] = 0;
            }
        }
        return DoubleStream.of(precision).sum() / (double) usedLabels.size();
    }

    private double calcWeightedAccuracy(int[][] confMatrix, int[] notPredicted, Set<String> usedLabels) {
        double[] accuracy = new double[notPredicted.length];
        for (int i = 0; i < notPredicted.length; i++) {
            int falsePositive = 0;
            for (int j = 0; j < notPredicted.length; j++) {
                if (j!= i) {
                    falsePositive += confMatrix[j][i];
                }
            }
            int falseNegative = 0;
            for (int j = 0; j < notPredicted.length; j++) {
                if (j!=i) {
                    falseNegative += confMatrix[i][j];
                }
            }
            int trueAll = 0;
            for (int j = 0; j < notPredicted.length; j++) {
                trueAll += confMatrix[j][j];
            }
            accuracy[i] = (double) trueAll / (double) (trueAll+falseNegative+falsePositive);
            if (Double.isNaN(accuracy[i])) {
                accuracy[i] = 0;
            }
        }
        return DoubleStream.of(accuracy).sum() / (double) usedLabels.size();
    }

    private int[][] addToConfMatrix(int[][] confMatrix, String correctLabel, String calculatedLabel, List<String> labels) {
        int indexCorrect = labels.indexOf(correctLabel);
        int indexCalc;
        if (correctLabel == calculatedLabel) {
            indexCalc = indexCorrect;
        } else {
            indexCalc = labels.indexOf(calculatedLabel);
        }
        confMatrix[indexCorrect][indexCalc] += 1;
        return confMatrix;
    }

    private static Pair<String, Double> getBestScoredLabel(List<Pair<String, Double>> scores) {
        Double bestScore = Double.MIN_VALUE;
        Pair<String, Double> bestPair = null;

        for(Pair<String, Double> score : scores) {
            if(score.getSecond() > bestScore) {
                bestScore = score.getSecond();
                bestPair = score;
            }
        }

        return bestPair;
    }
}
