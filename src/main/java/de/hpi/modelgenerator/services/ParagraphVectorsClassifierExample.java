package de.hpi.modelgenerator.services;


import de.hpi.modelgenerator.persistence.ShopOffer;
import de.hpi.modelgenerator.persistence.repo.OfferRepository;
import lombok.AccessLevel;
import lombok.Getter;
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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
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
public class ParagraphVectorsClassifierExample {

    private ParagraphVectors paragraphVectors;
    private LabelAwareIterator iterator;
    private TokenizerFactory tokenizerFactory;

    @Autowired
    private OfferRepository offerRepository;
    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsClassifierExample.class);
    private List<LabelledDocument> unlabelledOffers;


    public void makeParagraphVectors(long shopId)  throws Exception {
        List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
        List<LabelledDocument> labelledOffers = new ArrayList<>();
        setUnlabelledOffers(new ArrayList<>());

        int offerCount = offers.size();
        int offerIndex = 0;
        List<Integer> randoms = new ArrayList<>();

        log.info("Given a data set with " +  offerCount + " documents.");

        for(int i = 0; i < 0.3 * offerCount; i++) {
            boolean numberAlreadyTaken = true;
            int random = 0;
            while(numberAlreadyTaken) {
                random = ThreadLocalRandom.current().nextInt(0, offerCount + 1);
                numberAlreadyTaken = randoms.contains(random);
            }
            randoms.add(random);
        }

        for(ShopOffer offer : offers) {
            if(offer.getMappedCatalogCategory() == 0){
                offerIndex++;
                continue;
            }

            LabelledDocument document = new LabelledDocument();
            String title = "";
            String description = "";

            for(String key : offer.getTitles().keySet()) {
                title = offer.getTitles().get(key);
                break;
            }
            for(String key : offer.getDescriptions().keySet()) {
                description = offer.getDescriptions().get(key);
                break;
            }

            document.setContent(title + description);
            document.addLabel(Long.toString(offer.getMappedCatalogCategory()));

            if(randoms.contains(offerIndex)) {
                labelledOffers.add(document);
            } else {
                getUnlabelledOffers().add(document);
            }
            offerIndex++;
        }

        log.info("Use " +  labelledOffers.size() + " documents for training.");
        log.info("Use " +  getUnlabelledOffers().size() + " documents for validation.");

        iterator = new SimpleLabelAwareIterator(labelledOffers);
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

    public void checkUnlabeledData() throws Exception {
      /*
      At this point we assume that we have model built and we can check
      which categories our unlabeled document falls into.
      So we'll start loading our unlabeled documents and checking them
     */

        LabelAwareIterator unClassifiedIterator = new SimpleLabelAwareIterator(getUnlabelledOffers());

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

        while (unClassifiedIterator.hasNextDocument()) {
            LabelledDocument document = unClassifiedIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

         /*
          please note, document.getLabel() is used just to show which document we're looking at now,
          as a substitute for printing out the whole document name.
          So, labels on these two documents are used like titles,
          just to visualize our classification done properly
         */
            String bestLabel = getBestScoredLabel(scores);
            log.info("Document '" + document.getLabels() + "' falls into the following categories: " + bestLabel);
            for (Pair<String, Double> score: scores) {
                log.info("        " + score.getFirst() + ": " + score.getSecond());
            }

            if(bestLabel.equals(document.getLabels().get(0))){
                rightMatches++;
            } else {
                wrongMatches++;
            }
        }

        log.info("Right labels: " + rightMatches);
        log.info("Wrong labels: " + wrongMatches);

    }

    private String getBestScoredLabel(List<Pair<String, Double>> scores) {
        String bestLabel = "";
        Double bestScore = Double.MIN_VALUE;

        for(Pair<String, Double> score : scores) {
            if(score.getSecond() > bestScore) {
                bestScore = score.getSecond();
                bestLabel = score.getFirst();
            }
        }

        return bestLabel;
    }


}