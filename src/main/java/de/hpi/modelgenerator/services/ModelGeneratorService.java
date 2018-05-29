package de.hpi.modelgenerator.services;

import de.hpi.machinelearning.persistence.AttributeVector;
import de.hpi.machinelearning.persistence.FeatureInstance;
import de.hpi.machinelearning.persistence.LabeledModel;
import de.hpi.machinelearning.persistence.SerializedParagraphVectors;
import de.hpi.modelgenerator.persistence.ClassifierTrainingState;
import de.hpi.modelgenerator.persistence.MatchingResult;
import de.hpi.modelgenerator.persistence.ShopOffer;
import de.hpi.modelgenerator.persistence.repo.Cache;
import de.hpi.modelgenerator.persistence.repo.MatchingResultRepository;
import de.hpi.modelgenerator.persistence.repo.ModelRepository;
import de.hpi.modelgenerator.persistence.repo.OfferRepository;
import de.hpi.modelgenerator.properties.ModelGeneratorProperties;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static de.hpi.modelgenerator.services.MatchingModels.*;

@Service
@RequiredArgsConstructor
@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
@Slf4j
public class ModelGeneratorService {

    private static final String CATEGORY = "category";
    private static final String BRAND = "brand";

    private final OfferRepository offerRepository;

    private final ModelRepository modelRepository;

    private final ModelGeneratorProperties properties;

    private final MatchingResultRepository matchingResultRepository;

    private final Cache cache;

    private List<MatchingResult> trainingSet;

    private List<MatchingResult> testingSet;

    @Async("modelGeneratingThreadPoolTaskExecutor")
    public void generateCategoryClassifier(long shopId, ClassifierTrainingState state) throws IOException {
        List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
        List<LabelledDocument> documents = new LinkedList<>();

        List<Integer> numbers = IntStream.range(0, offers.size()).boxed().collect(Collectors.toCollection(LinkedList::new));
        int trainingSetSize = (int) (getProperties().getTestSetPercentage() * numbers.size());
        List<ShopOffer> training = IntStream.range(0, trainingSetSize).mapToObj(offers::get).collect(Collectors.toList());
        List<ShopOffer> testing = IntStream.range(trainingSetSize, numbers.size()).mapToObj(offers::get).collect(Collectors.toList());
        List<LabelledDocument> trainingSet = new LinkedList<>();
        List<LabelledDocument> testingSet = new LinkedList<>();

        for(ShopOffer offer : training) {
            if(offer.getTitles() != null && offer.getHigherLevelCategory() != null) {
                trainingSet.add(getLabelledDocumentByCategoryFromShopOffer(offer));
            }
        }

        for(ShopOffer offer : testing) {
            if(offer.getTitles() != null && offer.getHigherLevelCategory() != null) {
                testingSet.add(getLabelledDocumentByCategoryFromShopOffer(offer));
            }
        }

        ParagraphVectors paragraphVectors = NeuralNetClassifier.getParagraphVectors(trainingSet);
        getModelRepository().save(new SerializedParagraphVectors(paragraphVectors, CATEGORY));
        NeuralNetClassifier.checkUnlabeledData(paragraphVectors, testingSet);
        state.setCurrentlyLearning(false);
    }

    @Async("modelGeneratingThreadPoolTaskExecutor")
    public void generateBrandClassifier(long shopId, ClassifierTrainingState state) throws IOException {
        List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
        List<LabelledDocument> documents = new LinkedList<>();

        for(ShopOffer offer : offers) {
            if (offer.getTitles() != null && offer.getBrandName() != null) {
                documents.add(getLabelledDocumentByBrandFromShopOffer(offer));
            }
        }

        ParagraphVectors paragraphVectors = NeuralNetClassifier.getParagraphVectors(documents);
        getModelRepository().save(new SerializedParagraphVectors(paragraphVectors, BRAND));
        NeuralNetClassifier.checkUnlabeledData(paragraphVectors, documents);
        state.setCurrentlyLearning(false);
    }

    @Async("modelGeneratingThreadPoolTaskExecutor")
    public void generateModel(ClassifierTrainingState state){
        Map<String, LabeledModel> models = new HashMap<>();
        Map<Double, String> scoredModels = new HashMap<>();
        Instances trainingSet = createSet(10);

        models.put(MatchingModels.ADA_BOOST, getAdaBoost(trainingSet));
        models.put(MatchingModels.NAIVE_BAYES, getNaiveBayes(trainingSet));
        models.put(MatchingModels.LOGISTIC, getLogistic(trainingSet));
        models.put(MatchingModels.RANDOM_FOREST, getRandomForest(trainingSet));
        models.put(MatchingModels.K_NN, getKNN(trainingSet));
        models.put(MatchingModels.J48, getJ48(trainingSet));

        for(String key : models.keySet()) {
            double score = getClassificationError(models.get(key).getModel(), trainingSet);
            scoredModels.put(score, key);
        }

        Double bestModelScore = Collections.max(scoredModels.keySet());
        getModelRepository().save(models.get(scoredModels.get(bestModelScore)).toScoredModel(bestModelScore));
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

    public void generateRealCategoryClassifier(ClassifierTrainingState state) throws IOException {
        setTrainingAndTestingSet();
        List<LabelledDocument> trainingSet = getLabelledDocumentsByCategory(getTrainingSet());
        List<LabelledDocument> testingSet = getLabelledDocumentsByCategory(getTestingSet());

        log.info("Start generating category classifier at {} ", new Date());
        log.info("Use " + " documents for training.", trainingSet.size());
        log.info("Use " + " documents for validation.", testingSet.size());

        ParagraphVectors paragraphVectors = NeuralNetClassifier.getParagraphVectors(trainingSet);
        getModelRepository().save(new SerializedParagraphVectors(paragraphVectors, CATEGORY));
        NeuralNetClassifier.checkUnlabeledData(paragraphVectors, testingSet);
        state.setCurrentlyLearning(false);
        log.info("Successfully generated category classifier.");
    }

    public void generateRealBrandClassifier(ClassifierTrainingState state) throws IOException {
        setTrainingAndTestingSet();
        List<LabelledDocument> trainingSet = getLabelledDocumentsByBrand(getTrainingSet());
        List<LabelledDocument> testingSet = getLabelledDocumentsByBrand(getTestingSet());

        log.info("Start generating brand classifier at {} ", new Date());
        log.info("Use " + " documents for training.", trainingSet.size());
        log.info("Use " + " documents for validation.", testingSet.size());

        ParagraphVectors paragraphVectors = NeuralNetClassifier.getParagraphVectors(trainingSet);
        getModelRepository().save(new SerializedParagraphVectors(paragraphVectors, BRAND));
        NeuralNetClassifier.checkUnlabeledData(paragraphVectors, testingSet);
        state.setCurrentlyLearning(false);
        log.info("Successfully generated brand classifier.");
    }

    public void generateRealModel(ClassifierTrainingState state) {
        setTrainingAndTestingSet();
        Instances trainingSet = getInstances(getTrainingSet());
        Instances testingSet = getInstances(getTestingSet());
        Map<String, LabeledModel> models = new HashMap<>();
        Map<Double, String> scoredModels = new HashMap<>();

        log.info("Start generating model at {} ", new Date());
        log.info("Use " + " documents for training.", trainingSet.size());
        log.info("Use " + " documents for validation.", testingSet.size());

        models.put(MatchingModels.ADA_BOOST, getAdaBoost(trainingSet));
        models.put(MatchingModels.NAIVE_BAYES, getNaiveBayes(trainingSet));
        models.put(MatchingModels.LOGISTIC, getLogistic(trainingSet));
        models.put(MatchingModels.RANDOM_FOREST, getRandomForest(trainingSet));
        models.put(MatchingModels.K_NN, getKNN(trainingSet));
        models.put(MatchingModels.J48, getJ48(trainingSet));

        for(String key : models.keySet()) {
            double score = getClassificationError(models.get(key).getModel(), trainingSet);
            scoredModels.put(score, key);
        }

        Double leastClassificationError = Collections.min(scoredModels.keySet());
        getModelRepository().save(models.get(scoredModels.get(leastClassificationError)).toScoredModel(leastClassificationError));

        state.setCurrentlyLearning(false);
        log.info("Successfully generated model.");

    }

    private void setTrainingAndTestingSet() {
        if(getTrainingSet() == null || getTestingSet() == null) {
            List<MatchingResult> completeDataSet = new LinkedList<>();
            Set<Long> shopIds = getMatchingResultRepository().getShopIds();

            int matchesPerShop = Math.min(getProperties().getMatchesPerShop(),
                    getProperties().getMaximumMatchesForLearning() / shopIds.size());

            for (Long shopId : shopIds) {
                completeDataSet.addAll(getMatchingResultRepository().getMatches(shopId, matchesPerShop));
            }

            List<Integer> numbers = IntStream.range(0, completeDataSet.size()).boxed().collect(Collectors.toCollection(LinkedList::new));
            Collections.shuffle(numbers);
            int trainingSetSize = (int) (getProperties().getTestSetPercentage() * numbers.size());
            setTestingSet(IntStream.range(0, trainingSetSize).mapToObj(completeDataSet::get).collect(Collectors.toList()));
            setTestingSet(IntStream.range(trainingSetSize, numbers.size()).mapToObj(completeDataSet::get).collect(Collectors.toList()));
        }
    }

    private List<LabelledDocument> getLabelledDocumentsByCategory(List<MatchingResult> matchingResults) {
        List<LabelledDocument> documents = new LinkedList<>();
        for(MatchingResult matchingResult : matchingResults) {
            String category = matchingResult.getHigherLevelIdealoCategory();
            String title = matchingResult.getParsedData().getTitle();
            if(category != null && title != null) {
                documents.add(getLabelledDocument(title, category));
            }
        }

        return documents;
    }

    private List<LabelledDocument> getLabelledDocumentsByBrand(List<MatchingResult> matchingResults) {
        List<LabelledDocument> documents = new LinkedList<>();
        for(MatchingResult matchingResult : matchingResults) {
            String brand = matchingResult.getIdealoBrand();
            String title = matchingResult.getParsedData().getTitle();
            if(brand != null && title != null) {
                documents.add(getLabelledDocument(title, brand));
            }
        }

        return documents;
    }

    private LabelledDocument getLabelledDocument(String content, String label) {
        LabelledDocument document = new LabelledDocument();
        document.setContent(content);
        document.addLabel(label);
        return document;
    }

    private Instances getInstances(List<MatchingResult> matchingResults) {
        ArrayList<Attribute> features = new AttributeVector();
        Instances instanceSet = new Instances("Rel", features, matchingResults.size());

        List<Integer> numbers = IntStream.range(0, matchingResults.size()).boxed().collect(Collectors.toCollection(LinkedList::new));
        Collections.shuffle(numbers);

        // use 50% of results for matches
        for(int i = 0; i < numbers.size() / 2; i ++) {
            MatchingResult result = matchingResults.get(numbers.get(i));
            try {
                ShopOffer shopOffer = getCache().getOffer(result.getShopId(), result.getOfferKey());
                Instance instance = new FeatureInstance(shopOffer, result.getParsedData(), true);
                instanceSet.add(instance);
            } catch (HttpClientErrorException e) {
                e.printStackTrace();
            }
        }

        // use 50% of results for not-matches
        for(int i = numbers.size() / 2; i < numbers.size(); i++) {
            int nonMatchIndex = getDifferentRandom(i, 0, numbers.size());
            MatchingResult result = matchingResults.get(numbers.get(i));
            try {
                ShopOffer shopOffer = getCache().getOffer(result.getShopId(), matchingResults.get(nonMatchIndex).getOfferKey());
                Instance instance = new FeatureInstance(shopOffer, result.getParsedData(), false);
                instanceSet.add(instance);
            } catch (HttpClientErrorException e) {
                e.printStackTrace();
            }
        }

        return instanceSet;
    }

    private int getDifferentRandom(int excludedValue, int min, int max) {
        int random;
        do {
            random = ThreadLocalRandom.current().nextInt(min, max);
        } while(random == excludedValue);
        return random;
    }

}
