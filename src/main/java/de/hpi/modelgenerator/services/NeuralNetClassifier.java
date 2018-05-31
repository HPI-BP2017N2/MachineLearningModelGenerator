package de.hpi.modelgenerator.services;



import de.hpi.machinelearning.LabelSeeker;
import de.hpi.machinelearning.MeansBuilder;
import lombok.AccessLevel;
import lombok.Getter;
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

@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
@Slf4j
@Service
public class NeuralNetClassifier {

    ParagraphVectors getParagraphVectors(List<LabelledDocument> documents) {

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

         ParagraphVectors paragraphVectors = (new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(new SimpleLabelAwareIterator(documents))
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build());

        paragraphVectors.fit();
        return paragraphVectors;
    }

    void checkUnlabeledData(ParagraphVectors paragraphVectors, List<LabelledDocument> testingSet) {
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
        double labelThreshold = 0.5;

        Set<String> labels = new HashSet<>();

        for(LabelledDocument document : testingSet) {
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);


            Pair<String, Double> bestLabel = getBestScoredLabel(scores);
            if(bestLabel.getRight() < labelThreshold) {
                notLabeled++;
            } else if(bestLabel.getLeft().equals(document.getLabels().get(0))){
                rightMatches++;
            } else {
                wrongMatches++;
            }

            labels.add(document.getLabels().get(0));
        }


        log.info("Classification Error: {}", (double) wrongMatches / (double) (wrongMatches + rightMatches));
        log.info("Not labeled: {}", notLabeled);
        log.info("Different labels: {}", labels.size());

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
