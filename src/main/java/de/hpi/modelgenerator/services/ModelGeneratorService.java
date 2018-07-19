package de.hpi.modelgenerator.services;

import de.hpi.machinelearning.persistence.AttributeVector;
import de.hpi.machinelearning.persistence.FeatureInstance;
import de.hpi.machinelearning.persistence.LabeledModel;
import de.hpi.modelgenerator.persistence.ClassifierTrainingState;
import de.hpi.modelgenerator.persistence.MatchingResult;
import de.hpi.modelgenerator.persistence.ShopOffer;
import de.hpi.modelgenerator.persistence.repo.*;
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
import weka.classifiers.evaluation.Evaluation;
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

    private final ModelRepository modelRepository;
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
        getNeuralNetClassifier().checkUnlabeledData(paragraphVectors, testingSet, "category");
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
        getNeuralNetClassifier().checkUnlabeledData(paragraphVectors, testingSet, "brand");
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
    public void generateModel(ClassifierTrainingState state) throws Exception {
        if(state.isCurrentlyLearning()) {
            return;
        }

        if(!getModelRepository().brandClassifierExists() || !getModelRepository().categoryClassifierExists()) {
            throw new IllegalStateException("Brand classifier and Category classifier need to be generated first.");
        }

        getClassifier().loadBrandClassifier();
        getClassifier().loadCategoryClassifier();
        state.setCurrentlyLearning(true);
        setTrainingAndTestingSet();
        log.info("Start generating training set for model at {}", new Date());
        Instances trainingSet = getInstances(getTrainingSet());
        log.info("Finished generating training set for model at {}", new Date());
        log.info("Start generating testing set for model at {}", new Date());
        Instances testingSet = getInstances(getTestingSet());
        log.info("Finished generating testing set for model at {}", new Date());
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
            Evaluation eval = new Evaluation(trainingSet);
            eval.evaluateModel(model.getModel(), testingSet);
            //double score = getMatchingModels().getClassificationError(model.getModel(), trainingSet);
            //scoredModels.put(score, model.getModelType());
            System.out.println(eval.toSummaryString("\nResults "+model.getModelType()+"\n======\n", false));
            System.out.println("Weighted Precision: "+ String.valueOf(eval.weightedPrecision()));
            System.out.println("Weighted Recall: "+ String.valueOf(eval.weightedRecall()));
            System.out.println("Weighted FMeasure: "+ String.valueOf(eval.weightedFMeasure()));
            System.out.println("Weighted AreaUnderROC: "+ String.valueOf(eval.weightedAreaUnderROC()));
            System.out.println("Weighted MacroFMeasure: "+ String.valueOf(eval.unweightedMacroFmeasure()));
            System.out.println("Weighted MicroMeasure: "+ String.valueOf(eval.unweightedMicroFmeasure()));
            System.out.println("ConfusionMatrix: "+ Arrays.deepToString(eval.confusionMatrix()));
            System.out.println("falseNegative: "+ String.valueOf(eval.numFalseNegatives(0)));
            System.out.println("falsePositive: "+ String.valueOf(eval.numFalsePositives(0)));
            System.out.println("trueNegative: "+ String.valueOf(eval.numTrueNegatives(0)));
            System.out.println("truePositive: "+ String.valueOf(eval.numTruePositives(0)));

        }

        for(LabeledModel model : models) {
            Evaluation eval = new Evaluation(trainingSet);
            eval.evaluateModel(model.getModel(), testingSet);
            System.out.println(model.getModelType());
            System.out.println("ConfusionMatrix: " + Arrays.deepToString(eval.confusionMatrix()));
            for (double[] x : eval.confusionMatrix())
            {
                for (double y : x)
                {
                    System.out.print(String.valueOf(y) + " ");
                }
                System.out.println();
            }
        }
/*
        Double leastClassificationError = Collections.min(scoredModels.keySet());
        String bestScoredModelType = scoredModels.get(leastClassificationError);
        for(LabeledModel model : models) {
            if(model.getModelType().equals(bestScoredModelType)) {
                getModelRepository().save(model.toScoredModel(leastClassificationError));
                break;
            }
        }*/
        for(LabeledModel model : models) {
                getModelRepository().save(model.toScoredModel(1));
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
                documents.add(getLabelledDocument(title.toLowerCase(), category));
            }
        }

        return documents;
    }

    private List<LabelledDocument> getLabelledDocumentsByBrand(List<MatchingResult> matchingResults) {
        List<LabelledDocument> documents = new LinkedList<>();

        for(MatchingResult matchingResult : matchingResults) {
            try {
                String title = matchingResult.getParsedData().getTitle();
                if (title != null) {
                    if ( matchingResult.getIdealoBrand() != null) {
                        documents.add(getLabelledDocument(title, matchingResult.getIdealoBrand()));
                    } else {
                        ShopOffer shopOffer = getCache().getOffer(matchingResult.getShopId(), matchingResult.getOfferKey());
                        if (shopOffer != null && shopOffer.getBrandName() != null) {
                            documents.add(getLabelledDocument(title.toLowerCase(), shopOffer.getBrandName()));
                        }
                    }
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

        log.info("Start adding correct results to instance set at {}", new Date());
        // use 50% of results for matches
        Integer numbersSize = numbers.size() / 2;

        for(int i = 0; i < numbersSize; i ++) {
            MatchingResult result = matchingResults.get(numbers.get(i));
            try {
                ShopOffer shopOffer = getCache().getOffer(result.getShopId(), result.getOfferKey());
                if(shopOffer == null) continue;
                String brand = getBrand(result.getParsedData().getTitle());
                String category = getCategory(result.getParsedData().getTitle());
                Instance instance = new FeatureInstance(shopOffer, result.getParsedData(), true, brand, category);
                instanceSet.add(instance);

            } catch (HttpClientErrorException e) {
                e.printStackTrace();
            }
        }
        log.info("Finished adding correct results to instance set at {}", new Date());

        log.info("Start adding incorrect results to instance set based on same brand at {}", new Date());
        // use 50% of results for not-matches if brand is the same
        for (int j = 0; j <= 1; j++) {
            for (int i = numbersSize; i < numbers.size(); i++) {
                int nonMatchIndex = getDifferentRandom(i, 0, numbers.size());
                String matchBrand = matchingResults.get(numbers.get(i)).getIdealoBrand();
                for (MatchingResult result : matchingResults.subList(i - (numbersSize * j), numbers.size())) {
                    if (result.getIdealoBrand() == matchBrand) {
                        nonMatchIndex = matchingResults.indexOf(result);
                        break;
                    }
                }
                MatchingResult result = matchingResults.get(numbers.get(i));
                addResultToInstances(instanceSet, result, matchingResults, nonMatchIndex);
            }
            log.info("Finished adding incorrect results to instance set based on same brand at {}", new Date());

            log.info("Start adding incorrect results to instance set based on same category at {}", new Date());
            // use 50% of results for not-matches if category is the same
            for (int i = numbersSize; i < numbers.size(); i++) {
                int nonMatchIndex = getDifferentRandom(i, 0, numbers.size());
                String matchCategory = matchingResults.get(numbers.get(i)).getIdealoCategory();
                for (MatchingResult result : matchingResults.subList(i - (numbersSize * j), numbers.size())) {
                    if (result.getIdealoCategory() == matchCategory) {
                        nonMatchIndex = matchingResults.indexOf(result);
                        break;
                    }
                }
                MatchingResult result = matchingResults.get(numbers.get(i));
                addResultToInstances(instanceSet, result, matchingResults, nonMatchIndex);
            }
            log.info("Finished adding incorrect results to instance set based on same category at {}", new Date());
        }

        instanceSet.setClassIndex(10);
        return instanceSet;
    }

    private void addResultToInstances(Instances instanceSet, MatchingResult result, List<MatchingResult> matchingResults, int nonMatchIndex) {
        try {
            ShopOffer shopOffer = getCache().getOffer(matchingResults.get(nonMatchIndex).getShopId(), matchingResults.get(nonMatchIndex).getOfferKey());
            if(shopOffer != null) {
                String brand = getBrand(result.getParsedData().getTitle());
                String category = getCategory(result.getParsedData().getTitle());
                Instance instance = new FeatureInstance(shopOffer, result.getParsedData(), false, brand, category);
                instanceSet.add(instance);
            }
        } catch (HttpClientErrorException e) {
            e.printStackTrace();
        }
    }

    private String getCategory(String offerTitle) {
        if(offerTitle != null) {
            Pair<String, Double> pair = getClassifier().getCategory(offerTitle.toLowerCase());
            if(pair == null) {
                return null;
            }
            return pair.getRight() < getProperties().getLabelThresholdCategory() ? null : pair.getLeft();
        }

        return null;
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
            Pair<String, Double> pair = getClassifier().getBrand(offerTitle.toLowerCase());
            if(pair == null) {
                return null;
            }
            return pair.getRight() < getProperties().getLabelThresholdBrand() ? null : pair.getLeft();
        }

        return null;
    }


}
