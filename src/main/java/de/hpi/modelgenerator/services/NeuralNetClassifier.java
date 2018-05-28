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
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.SimpleLabelAwareIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
@Slf4j
public class NeuralNetClassifier {

    private ParagraphVectors paragraphVectors;
    private TokenizerFactory tokenizerFactory;
    private List<LabelledDocument> unlabeledOffers;

    public ParagraphVectors getParagraphVectors(List<LabelledDocument> documents) {
        List<LabelledDocument> labeledOffers = new LinkedList<>();
        setUnlabeledOffers(new LinkedList<>());
        List<Integer> randoms = getRandomIntegers(documents.size(),  (int)(0.9 * documents.size()));
        int offerIndex = 0;

        log.info("Given a data set with " +  documents.size() + " documents.");

        for(LabelledDocument document: documents) {
            if (randoms.contains(offerIndex)) {
                labeledOffers.add(document);
            } else {
                getUnlabeledOffers().add(document);
            }
            offerIndex++;
        }

        log.info("Use " + labeledOffers.size() + " documents for training.");
        log.info("Use " + getUnlabeledOffers().size() + " documents for validation.");

        setTokenizerFactory(new DefaultTokenizerFactory());
        getTokenizerFactory().setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        setParagraphVectors(new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(new SimpleLabelAwareIterator(labeledOffers))
                .trainWordVectors(true)
                .tokenizerFactory(getTokenizerFactory())
                .build());

        // Start model training
        getParagraphVectors().fit();
        log.info("DONE BUILDING MODEL");

        return getParagraphVectors();
    }

    public void checkUnlabeledData() {
        LabelAwareIterator unClassifiedIterator = new SimpleLabelAwareIterator(getUnlabeledOffers());
        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)getParagraphVectors().getLookupTable(),
                getTokenizerFactory());
        LabelSeeker seeker = new LabelSeeker(getParagraphVectors().getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) getParagraphVectors().getLookupTable());
        int rightMatches = 0;
        int wrongMatches = 0;
        int notLabeled = 0;
        double labelThreshold = 0.5;

        Set<String> labels = new HashSet<>();

        while (unClassifiedIterator.hasNextDocument()) {
            LabelledDocument document = unClassifiedIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);


            String bestLabel = getBestScoredLabel(scores);
            if(getBestScore(scores) < labelThreshold) {
                notLabeled++;
            } else if(bestLabel.equals(document.getLabels().get(0))){
                rightMatches++;
            } else {
                wrongMatches++;
            }

            labels.add(document.getLabels().get(0));
        }

        log.info("Right labels: " + rightMatches);
        log.info("Wrong labels: " + wrongMatches);
        log.info("Not labeled: " + notLabeled);
        log.info("Different labels: " + labels.size());

    }

    private String getBestScoredLabel(List<Pair<String, Double>> scores) {
        String bestLabel = null;
        Double bestScore = Double.MIN_VALUE;

        for(Pair<String, Double> score : scores) {
            if(score.getSecond() > bestScore) {
                bestScore = score.getSecond();
                bestLabel = score.getFirst();
            }
        }

        return bestLabel;
    }

    private Double getBestScore(List<Pair<String, Double>> scores) {
        Double bestScore = Double.MIN_VALUE;

        for(Pair<String, Double> score : scores) {
            if(score.getSecond() > bestScore) {
                bestScore = score.getSecond();
            }
        }

        return bestScore;
    }

    private List<Integer> getRandomIntegers(int range, int numberOfRandoms) {
        List<Integer> randoms = new ArrayList<>();
        for(int i = 0; i < numberOfRandoms; i++) {
            boolean numberAlreadyTaken = true;
            int random = 0;
            while(numberAlreadyTaken) {
                random = ThreadLocalRandom.current().nextInt(0, range + 1);
                numberAlreadyTaken = randoms.contains(random);
            }
            randoms.add(random);
        }

        return randoms;
    }

}
