package de.hpi.modelgenerator.services;

import de.hpi.machinelearning.LabelSeeker;
import de.hpi.machinelearning.MeansBuilder;
import de.hpi.modelgenerator.persistence.repo.ModelRepository;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;

@Service
@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
@Slf4j
@RequiredArgsConstructor
class ProbabilityClassifier {

    private final ModelRepository modelRepository;
    private final TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

    private ParagraphVectors brandClassifier;
    private MeansBuilder brandMeansBuilder;
    private LabelSeeker brandLabelSeeker;

    Pair<String, Double> getBrand(String offerTitle) {
        LabelledDocument document = getLabelledDocumentFromTitle(offerTitle);
        INDArray documentAsCentroid = getBrandMeansBuilder().documentAsVector(document);
        List<Pair<String, Double>> scores = getBrandLabelSeeker().getScores(documentAsCentroid);

        return getBestScoredLabel(scores);
    }

    void loadBrandClassifier() throws IOException {
        setBrandClassifier(getModelRepository().getBrandClassifier());
        setBrandMeansBuilder(new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)getBrandClassifier().getLookupTable(),
                getTokenizerFactory()));
        setBrandLabelSeeker(new LabelSeeker(getBrandClassifier().getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) getBrandClassifier().getLookupTable()));

        log.info("Loaded brand classifier");
    }

    private LabelledDocument getLabelledDocumentFromTitle(String offerTitle) {
        LabelledDocument document = new LabelledDocument();
        document.setContent(offerTitle);
        return document;
    }

    private Pair<String, Double> getBestScoredLabel(List<Pair<String, Double>> scores) {
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
