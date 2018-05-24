package de.hpi.modelgenerator.services;

import de.hpi.modelgenerator.persistence.*;
import de.hpi.modelgenerator.persistence.repo.ModelRepository;
import de.hpi.modelgenerator.persistence.repo.OfferRepository;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import weka.core.Instances;

import java.util.*;

import static de.hpi.modelgenerator.services.MatchingModels.*;

@Service
@RequiredArgsConstructor
@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
public class ModelGeneratorService {

    private final OfferRepository offerRepository;

    private final ModelRepository modelRepository;


    @Async
    public void generateCategoryClassifier(long shopId, ClassifierTrainingState state) {
        List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
        List<LabelledDocument> documents = new LinkedList<>();

        for(ShopOffer offer : offers) {
            if(offer.getTitles() != null && offer.getHigherLevelCategory() != null) {
                documents.add(getLabelledDocumentByCategoryFromShopOffer(offer));
            }
        }

        NeuralNetClassifier classifier = new NeuralNetClassifier();
        getModelRepository().save(new SerializedParagraphVectors(classifier.getParagraphVectors(documents)));
        classifier.checkUnlabeledData();
        state.setCurrentlyLearning(false);
    }

    @Async
    public void generateBrandClassifier(long shopId, ClassifierTrainingState state) {
        List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
        List<LabelledDocument> documents = new LinkedList<>();

        for(ShopOffer offer : offers) {
            if (offer.getTitles() != null && offer.getBrandName() != null) {
                documents.add(getLabelledDocumentByBrandFromShopOffer(offer));
            }
        }

        NeuralNetClassifier classifier = new NeuralNetClassifier();
        getModelRepository().save(new SerializedParagraphVectors(classifier.getParagraphVectors(documents)));
        classifier.checkUnlabeledData();
        state.setCurrentlyLearning(false);
    }

    @Async
    public void generateModel(ClassifierTrainingState state){
        List<LabeledModel> models = new ArrayList<>();
        Map<Double, LabeledModel> scoredModels = new HashMap<>();
        Instances trainingSet = createSet(10);

        models.add(getAdaBoost(trainingSet));
        models.add(getNaiveBayes(trainingSet));
        models.add(getLogistic(trainingSet));
        models.add(getRandomForest(trainingSet));
        models.add(getKNN(trainingSet));
        models.add(getJ48(trainingSet));

        for(LabeledModel model : models) {
            double score = evaluateModel(model.getModel(), trainingSet);
            scoredModels.put(score, model);
        }

        Double bestModelScore = Collections.max(scoredModels.keySet());
        getModelRepository().save(scoredModels.get(bestModelScore).toScoredModel(bestModelScore));
        state.setCurrentlyLearning(false);
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
