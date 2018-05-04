package de.hpi.modelgenerator.services;


import de.hpi.modelgenerator.persistence.ShopOffer;
import de.hpi.modelgenerator.persistence.repo.OfferRepository;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * This is basic example for documents classification done with DL4j ParagraphVectors.
 * The overall idea is to use ParagraphVectors in the same way we use LDA:
 * topic space modelling.
 *
 * In this example we assume we have few labeled categories that we can use
 * for training, and few unlabeled documents. And our goal is to determine,
 * which category these unlabeled documents fall into
 *
 *
 * Please note: This example could be improved by using learning cascade
 * for higher accuracy, but that's beyond basic example paradigm.
 *
 * @author raver119@gmail.com
 */
@Service
@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
@RequiredArgsConstructor
public class ParagraphVectorsClassifierExample {

    private ParagraphVectors paragraphVectors;
    private LabelAwareIterator iterator;
    private TokenizerFactory tokenizerFactory;
    private List<LabelledDocument> unlabeledOffers;
    private long currentShopId;

    private final OfferRepository offerRepository;
    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsClassifierExample.class);


    public void makeParagraphVectors(long shopId) {
        setCurrentShopId(shopId);
        List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
        List<LabelledDocument> labeledOffers = new LinkedList<>();
        setUnlabeledOffers(new LinkedList<>());
        List<Integer> randoms = getRandomIntegers(offers.size(),  (int)(0.3 * offers.size()));
        int offerIndex = 0;

        log.info("Given a data set with " +  offers.size() + " documents.");

        for(ShopOffer offer : offers) {
            if(offer.getTitles() == null && offer.getDescriptions() == null){
                offerIndex++;
                continue;
            }

            LabelledDocument document = getLabelledDocumentFromShopOffer(offer);

            if(randoms.contains(offerIndex)) {
                labeledOffers.add(document);
            } else {
                getUnlabeledOffers().add(document);
            }
            offerIndex++;
        }

        log.info("Use " + labeledOffers.size() + " documents for training.");
        log.info("Use " + getUnlabeledOffers().size() + " documents for validation.");

        iterator = new SimpleLabelAwareIterator(labeledOffers);
        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        paragraphVectors = new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();

        // Start model training
        paragraphVectors.fit();

        log.info("DONE BUILDING MODEL");
    }

    public void checkUnlabeledData(long shopId) throws IllegalStateException {
        if(shopId != getCurrentShopId()) {
            throw new IllegalStateException();
        }

      /*
      At this point we assume that we have model built and we can check
      which categories our unlabeled document falls into.
      So we'll start loading our unlabeled documents and checking them
     */

      LabelAwareIterator unClassifiedIterator = new SimpleLabelAwareIterator(getUnlabeledOffers());

     /*
      Now we'll iterate over unlabeled data, and check which label it could be assigned to
      Please note: for many domains it's normal to have 1 document fall into few labels at once,
      with different "weight" for each.
     */
        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        int rightMatches = 0;
        int wrongMatches = 0;
        int notLabeled = 0;
        double labelThreshold = 0.5;

        Set<String> labels = new HashSet<>();

        while (unClassifiedIterator.hasNextDocument()) {
            LabelledDocument document = unClassifiedIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            //log.info("Document '" + document.getLabels() + "' falls into the following categories: " );
            /*for (Pair<String, Double> score: scores) {
                log.info("        " + score.getFirst() + ": " + score.getSecond());
            }*/

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

    private LabelledDocument getLabelledDocumentFromShopOffer(ShopOffer offer) {
        LabelledDocument document = new LabelledDocument();
        String title = null;
        String description = null;

        if(offer.getTitles() != null) {
            title = offer.getTitles().get(offer.getTitles().keySet().iterator().next());
        }


        if(offer.getDescriptions() != null) {
            description = offer.getDescriptions().get(offer.getDescriptions().keySet().iterator().next());
        }

        document.setContent(title + description);
        document.addLabel(Long.toString(offer.getMappedCatalogCategory()));

        return document;
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