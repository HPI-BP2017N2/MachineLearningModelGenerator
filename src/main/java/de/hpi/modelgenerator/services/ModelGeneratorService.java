package de.hpi.modelgenerator.services;

import de.hpi.modelgenerator.persistence.ScoredModel;
import de.hpi.modelgenerator.persistence.LabeledModel;
import de.hpi.modelgenerator.persistence.ShopOffer;
import de.hpi.modelgenerator.persistence.repo.OfferRepository;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.springframework.stereotype.Service;
import weka.core.Instances;

import java.util.*;

import static de.hpi.modelgenerator.services.MatchingModels.*;

@Service
@RequiredArgsConstructor
@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
public class ModelGeneratorService {

    private ParagraphVectors categoryClassifier;

    private ParagraphVectors brandClassifier;

    private LabeledModel mlModel;

    private double modelScore;

    private final OfferRepository offerRepository;

    public ParagraphVectors getCategoryClassifier(long shopId) {
        if(getCategoryClassifier() == null) {
            List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
            List<LabelledDocument> documents = new LinkedList<>();

            for(ShopOffer offer : offers) {
                if(offer.getTitles() != null && offer.getHigherLevelCategory() != null) {
                    documents.add(getLabelledDocumentByCategoryFromShopOffer(offer));
                }
            }

            NeuralNetClassifier classifier = new NeuralNetClassifier();
            setCategoryClassifier(classifier.getParagraphVectors(documents));
            classifier.checkUnlabeledData();
        }

        return getCategoryClassifier();


    }

    public ParagraphVectors getBrandClassifier(long shopId) {
        if(getBrandClassifier() == null) {
            List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
            List<LabelledDocument> documents = new LinkedList<>();

            for(ShopOffer offer : offers) {
                if(offer.getTitles() != null && offer.getBrandName() != null) {
                    documents.add(getLabelledDocumentByBrandFromShopOffer(offer));
                }
            }

            NeuralNetClassifier classifier = new NeuralNetClassifier();
            setBrandClassifier(classifier.getParagraphVectors(documents));
            classifier.checkUnlabeledData();
        }

        return  getBrandClassifier();


    }

    public ScoredModel getModel(){
        if(getMlModel() == null) {
            setMlModel(getBestScoredModel());
        }

        return getMlModel().toScoredModel(getModelScore());

    }

    private LabeledModel getBestScoredModel(){
        List<LabeledModel> models = new ArrayList<>();
        Map<Double, LabeledModel> scoredModels = new HashMap<>();
        Instances trainingSet = createSet(10);

        models.add(getAdaBoost(trainingSet));
        models.add(getNaiveBayes(trainingSet));
        models.add(getLogistic(trainingSet));
        models.add(getRandomForest(trainingSet));
        models.add(getKNN(trainingSet));
        //models.add(getLinearRegression(trainingSet));
        models.add(getJ48(trainingSet));

        for(LabeledModel model : models) {
            double score = evaluateModel(model.getModel(), trainingSet);
            scoredModels.put(score, model);
        }

        setModelScore(Collections.max(scoredModels.keySet()));
        return scoredModels.get(getModelScore());
    }

    private LabelledDocument getLabelledDocumentByCategoryFromShopOffer(ShopOffer offer) {
        LabelledDocument document = new LabelledDocument();
        String title = null;

        if(offer.getTitles() != null) {
            title = offer.getTitles().get(offer.getTitles().keySet().iterator().next());
        }

        document.setContent(title);
        document.addLabel(offer.getHigherLevelCategory());

        return document;
    }

    private LabelledDocument getLabelledDocumentByBrandFromShopOffer(ShopOffer offer) {
        LabelledDocument document = new LabelledDocument();
        String title = null;

        if(offer.getTitles() != null) {
            title = offer.getTitles().get(offer.getTitles().keySet().iterator().next());
        }

        document.setContent(title);
        document.addLabel(offer.getBrandName());

        return document;
    }

}
