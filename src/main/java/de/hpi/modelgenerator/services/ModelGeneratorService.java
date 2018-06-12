package de.hpi.modelgenerator.services;

import de.hpi.machinelearning.persistence.AttributeVector;
import de.hpi.machinelearning.persistence.FeatureInstance;
import de.hpi.machinelearning.persistence.LabeledModel;
import de.hpi.modelgenerator.persistence.ClassifierTrainingState;
import de.hpi.modelgenerator.persistence.MatchingResult;
import de.hpi.modelgenerator.persistence.ShopOffer;
import de.hpi.modelgenerator.persistence.repo.Cache;
import de.hpi.modelgenerator.persistence.repo.MatchingResultRepository;
import de.hpi.modelgenerator.persistence.repo.ModelFileRepository;
import de.hpi.modelgenerator.persistence.repo.ModelMongoRepository;
import de.hpi.modelgenerator.properties.ModelGeneratorProperties;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.nd4j.linalg.primitives.Pair;
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

@Service
@RequiredArgsConstructor
@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
@Slf4j
public class ModelGeneratorService {

    private static final String CATEGORY = "category";
    private static final String BRAND = "brand";

    private final ModelMongoRepository modelRepository;
    private final ModelGeneratorProperties properties;
    private final MatchingResultRepository matchingResultRepository;
    private final Cache cache;
    private final NeuralNetClassifier neuralNetClassifier;
    private final MatchingModels matchingModels;
    private final ProbabilityClassifier classifier;

    private List<MatchingResult> trainingSet;
    private List<MatchingResult> testingSet;

    /**
     * This method generates and saves a neural network for labelling the category of an offer.
     * If necessary, the training set will be created.
     * @param state Mutex to avoid multiple generation of classifier
     * @throws IOException when classifier cannot be serialized
     */
    public void generateCategoryClassifier(ClassifierTrainingState state) throws IOException {
        if(state.isCurrentlyLearning()) {
            return;
        }

        state.setCurrentlyLearning(true);
        setTrainingAndTestingSet();
        List<LabelledDocument> trainingSet = getLabelledDocumentsByCategory(getTrainingSet());
        List<LabelledDocument> testingSet = getLabelledDocumentsByCategory(getTestingSet());

        log.info("Start generating category classifier at {} ", new Date());
        log.info("Use {} documents for training.", trainingSet.size());
        log.info("Use {} documents for validation.", testingSet.size());

        ParagraphVectors paragraphVectors = getNeuralNetClassifier().getParagraphVectors(trainingSet);
        state.setCurrentlyLearning(false);
        getModelRepository().save(paragraphVectors, CATEGORY);
        log.info("Successfully generated category classifier.");
    }

    /**
     * This method generates and saves a neural network for labelling the brand of an offer.
     * If necessary, the training set will be created.
     * @param state Mutex to avoid multiple generation of classifier
     * @throws IOException when classifier cannot be serialized
     */
    public void generateBrandClassifier(ClassifierTrainingState state) throws IOException {
        if(state.isCurrentlyLearning()) {
            return;
        }

        state.setCurrentlyLearning(true);
        setTrainingAndTestingSet();
        List<LabelledDocument> trainingSet = getLabelledDocumentsByBrand(getTrainingSet());
        List<LabelledDocument> testingSet = getLabelledDocumentsByBrand(getTestingSet());

        log.info("Start generating brand classifier at {} ", new Date());
        log.info("Use {} documents for training.", trainingSet.size());
        log.info("Use {} documents for validation.", testingSet.size());

        ParagraphVectors paragraphVectors = getNeuralNetClassifier().getParagraphVectors(trainingSet);
        state.setCurrentlyLearning(false);
        getModelRepository().save(paragraphVectors, BRAND);
        log.info("Successfully generated brand classifier.");
    }

    /**
     * This method generates and saves a model for classifying whether two offers match or not.
     * Multiple classifiers will be trained and the best one (lowest classification error) is chosen.
     * If necessary, the training and testing sets set will be created.
     * This method needs the brand classifier to be generated first, since it is necessary for a feature.
     * @param state Mutex to avoid multiple generation of classifier
     * @throws IllegalStateException when brand classifier is not present
     * @throws IOException when brand classifier cannot be deserialized
     */
    public void generateModel(ClassifierTrainingState state) throws IllegalStateException, IOException {
        if(state.isCurrentlyLearning()) {
            return;
        }

        if(!getModelRepository().brandClassifierExists()) {
            throw new IllegalStateException("Brand classifier needs to be generated first.");
        }

        getClassifier().loadBrandClassifier();
        state.setCurrentlyLearning(true);
        setTrainingAndTestingSet();
        log.info("Start generating training and testing set for model at {}", new Date());
        Instances trainingSet = getInstances(getTrainingSet());
        Instances testingSet = getInstances(getTestingSet());
        log.info("Finished generating training and testing set for model at {}", new Date());
        List<LabeledModel> models = new LinkedList<>();
        Map<Double, String> scoredModels = new HashMap<>();

        log.info("Start generating model at {} ", new Date());
        log.info("Use {} documents for training.", trainingSet.size());
        log.info("Use {} documents for validation.", testingSet.size());

        models.add(getMatchingModels().getAdaBoost(trainingSet));
        models.add(getMatchingModels().getNaiveBayes(trainingSet));
        models.add(getMatchingModels().getLogistic(trainingSet));
        models.add(getMatchingModels().getRandomForest(trainingSet));
        models.add(getMatchingModels().getKNN(trainingSet));
        models.add(getMatchingModels().getJ48(trainingSet));

        for(LabeledModel model : models) {
            double score = getMatchingModels().getClassificationError(model.getModel(), trainingSet);
            scoredModels.put(score, model.getModelType());
        }

        Double leastClassificationError = Collections.min(scoredModels.keySet());
        String bestScoredModelType = scoredModels.get(leastClassificationError);
        for(LabeledModel model : models) {
            if(model.getModelType().equals(bestScoredModelType)) {
                getModelRepository().save(model.toScoredModel(leastClassificationError));
                break;
            }
        }

        state.setCurrentlyLearning(false);
        log.info("Successfully generated model.");

    }

    /**
     * This method deletes training and testing set.
     */
    public void freeTestingSet() {
        setTestingSet(null);
        setTrainingSet(null);
        System.gc();
    }

    /**
     * This method deletes training and testing set and loads them again.
     */
    public void refreshTrainingAndTestingSet() {
        freeTestingSet();
        setTrainingAndTestingSet();
    }

    void setTrainingAndTestingSet() {
        if(!trainingAndTestingSetExist()) {
            log.info("Start loading training and testing set at {}", new Date());
            List<MatchingResult> completeDataSet = new LinkedList<>();
            Set<Long> shopIds = getMatchingResultRepository().getShopIds();

            int matchesPerShop = Math.min(getProperties().getMatchesPerShop(),
                    getProperties().getMaximumMatchesForLearning() / shopIds.size());

            for (Long shopId : shopIds) {
                completeDataSet.addAll(getMatchingResultRepository().getMatches(shopId, matchesPerShop));
            }

            List<Integer> numbers = IntStream.range(0, completeDataSet.size()).boxed().collect(Collectors.toCollection(LinkedList::new));
            Collections.shuffle(numbers);
            int trainingSetSize = (int) (getProperties().getTrainingSetPercentage() * numbers.size());
            setTrainingSet(IntStream.range(0, trainingSetSize).mapToObj(completeDataSet::get).collect(Collectors.toList()));
            setTestingSet(IntStream.range(trainingSetSize, numbers.size()).mapToObj(completeDataSet::get).collect(Collectors.toList()));
            log.info("Finished loading training and testing set at {}", new Date());
        }
    }

    private boolean trainingAndTestingSetExist() {
        return getTrainingSet() != null && getTestingSet() != null;
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
            try {
                ShopOffer shopOffer = getCache().getOffer(matchingResult.getShopId(), matchingResult.getOfferKey());
                String title = matchingResult.getParsedData().getTitle();
                if (shopOffer != null && title != null && shopOffer.getBrandName() != null) {
                    documents.add(getLabelledDocument(title, shopOffer.getBrandName()));
                }
            } catch (HttpClientErrorException e) {
                e.printStackTrace();
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
                if(shopOffer == null) continue;
                String brand = getBrand(result.getParsedData().getTitle());
                Instance instance = new FeatureInstance(shopOffer, result.getParsedData(), true, brand);
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
                if(shopOffer != null) {
                    String brand = getBrand(result.getParsedData().getTitle());
                    Instance instance = new FeatureInstance(shopOffer, result.getParsedData(), false, brand);
                    instanceSet.add(instance);
                }
            } catch (HttpClientErrorException e) {
                e.printStackTrace();
            }
        }

        instanceSet.setClassIndex(12);
        return instanceSet;
    }

    private static int getDifferentRandom(int excludedValue, int min, int max) {
        if(excludedValue == min || excludedValue == max) return excludedValue;

        int random;
        do {
            random = ThreadLocalRandom.current().nextInt(min, max);
        } while(random == excludedValue);
        return random;
    }

    private String getBrand(String offerTitle) {
        if(offerTitle != null) {
            Pair<String, Double> pair = getClassifier().getBrand(offerTitle);
            if(pair == null) {
                return null;
            }
            return pair.getRight() < getProperties().getLabelThreshold() ? null : pair.getLeft();
        }

        return null;
    }


}
