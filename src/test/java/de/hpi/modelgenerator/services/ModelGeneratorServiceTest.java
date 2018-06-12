package de.hpi.modelgenerator.services;

import de.hpi.machinelearning.persistence.LabeledModel;
import de.hpi.machinelearning.persistence.ScoredModel;
import de.hpi.modelgenerator.persistence.ClassifierTrainingState;
import de.hpi.modelgenerator.persistence.MatchingResult;
import de.hpi.modelgenerator.persistence.ParsedOffer;
import de.hpi.modelgenerator.persistence.repo.Cache;
import de.hpi.modelgenerator.persistence.repo.MatchingResultRepository;
import de.hpi.modelgenerator.persistence.repo.ModelFileRepository;
import de.hpi.modelgenerator.persistence.repo.ModelMongoRepository;
import de.hpi.modelgenerator.properties.ModelGeneratorProperties;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

import java.io.IOException;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.eq;
import static org.mockito.MockitoAnnotations.initMocks;

@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
public class ModelGeneratorServiceTest {

    @Getter(AccessLevel.PRIVATE) private final static long EXAMPLE_SHOP_ID = 1234L;
    @Getter(AccessLevel.PRIVATE) private final static int EXAMPLE_RESULTS_PER_SHOP = 2;
    @Getter(AccessLevel.PRIVATE) private final static int EXAMPLE_MAX_RESULTS = 2;
    @Getter(AccessLevel.PRIVATE) private final static double EXAMPLE_TEST_SET_PERCENTAGE = 0.5;
    @Getter(AccessLevel.PRIVATE) private final static String EXAMPLE_TITLE = "iPhone7";
    @Getter(AccessLevel.PRIVATE) private final static String EXAMPLE_BRAND = "Apple";
    @Getter(AccessLevel.PRIVATE) private final static String EXAMPLE_CATEGORY = "1234";
    @Getter(AccessLevel.PRIVATE) private final static String EXAMPLE_OFFER_KEY = "1234";
    @Getter(AccessLevel.PRIVATE) private final static LabeledModel EXAMPLE_MODEL = new LabeledModel(new AdaBoostM1(), "example");
    public static final String BRAND = "brand";
    public static final String CATEGORY = "category";


    private final List<MatchingResult> exampleMatchingResults = new LinkedList<>();

    @Mock private MatchingResultRepository matchingResultRepository;
    @Mock private ModelMongoRepository modelRepository;
    @Mock private Cache cache;
    @Mock private ModelGeneratorProperties properties;
    @Mock private MatchingModels matchingModels;
    @Mock private ProbabilityClassifier probabilityClassifier;
    @Mock private NeuralNetClassifier neuralNetClassifier;
    @Mock private ParagraphVectors paragraphVectors;
    @Mock private Classifier classifier;
    @Mock private ClassifierTrainingState state;

    private ModelGeneratorService service;

    @Before
    public void setup() {
        initMocks(this);

        setService(new ModelGeneratorService(
                getModelRepository(),
                getProperties(),
                getMatchingResultRepository(),
                getCache(),
                getNeuralNetClassifier(),
                getMatchingModels(),
                getProbabilityClassifier()
        ));

        ParsedOffer parsedOffer = new ParsedOffer();
        parsedOffer.setTitle(getEXAMPLE_TITLE());
        MatchingResult matchingResult = new MatchingResult();
        matchingResult.setHigherLevelIdealoCategory(getEXAMPLE_CATEGORY());
        matchingResult.setIdealoBrand(getEXAMPLE_BRAND());
        matchingResult.setParsedData(parsedOffer);
        matchingResult.setOfferKey(getEXAMPLE_OFFER_KEY());
        getExampleMatchingResults().add(matchingResult);
        getExampleMatchingResults().add(matchingResult);

        Set<Long> shopIds = new HashSet<>();
        shopIds.add(getEXAMPLE_SHOP_ID());

        doReturn(getEXAMPLE_MAX_RESULTS()).when(getProperties()).getMaximumMatchesForLearning();
        doReturn(getEXAMPLE_RESULTS_PER_SHOP()).when(getProperties()).getMatchesPerShop();
        doReturn(getEXAMPLE_TEST_SET_PERCENTAGE()).when(getProperties()).getTrainingSetPercentage();
        doReturn(shopIds).when(getMatchingResultRepository()).getShopIds();
        doReturn(getExampleMatchingResults()).when(getMatchingResultRepository()).getMatches(anyLong(), anyInt());
    }


    @Test
    public void setTrainingAndTestingSet() {
        getService().setTrainingAndTestingSet();

        verify(getMatchingResultRepository()).getShopIds();
        verify(getMatchingResultRepository(), times(1)).getMatches(anyLong(), anyInt());
    }

    @Test
    public void generateCategoryClassifier() throws IOException {
        doReturn(getParagraphVectors()).when(getNeuralNetClassifier()).getParagraphVectors(anyList());
        doNothing().when(getModelRepository()).save(any(ParagraphVectors.class), eq(CATEGORY));

        getService().generateCategoryClassifier(getState());

        verify(getNeuralNetClassifier()).getParagraphVectors(anyList());
        verify(getModelRepository()).save(any(ParagraphVectors.class), eq(CATEGORY));
        verify(getState()).isCurrentlyLearning();
        verify(getState()).setCurrentlyLearning(true);
        verify(getState()).setCurrentlyLearning(false);
    }

    @Test
    public void generateBrandClassifier() throws IOException {
        doReturn(getParagraphVectors()).when(getNeuralNetClassifier()).getParagraphVectors(anyList());
        doNothing().when(getModelRepository()).save(any(ParagraphVectors.class), eq(BRAND));

        getService().generateBrandClassifier(getState());

        verify(getNeuralNetClassifier()).getParagraphVectors(anyList());
        verify(getModelRepository()).save(any(ParagraphVectors.class), eq(BRAND));
        verify(getState()).isCurrentlyLearning();
        verify(getState()).setCurrentlyLearning(true);
        verify(getState()).setCurrentlyLearning(false);
    }

    @Test
    public void generateModel() throws IOException {
        doReturn(getParagraphVectors()).when(getNeuralNetClassifier()).getParagraphVectors(anyList());
        doReturn(true).when(getModelRepository()).brandClassifierExists();
        doNothing().when(getModelRepository()).save(any(ScoredModel.class));


        doReturn(getEXAMPLE_MODEL()).when(getMatchingModels()).getLogistic(any(Instances.class));
        doReturn(getEXAMPLE_MODEL()).when(getMatchingModels()).getAdaBoost(any(Instances.class));
        doReturn(getEXAMPLE_MODEL()).when(getMatchingModels()).getJ48(any(Instances.class));
        doReturn(getEXAMPLE_MODEL()).when(getMatchingModels()).getKNN(any(Instances.class));
        doReturn(getEXAMPLE_MODEL()).when(getMatchingModels()).getNaiveBayes(any(Instances.class));
        doReturn(getEXAMPLE_MODEL()).when(getMatchingModels()).getRandomForest(any(Instances.class));
        doReturn(null).when(getCache()).getOffer(anyLong(), anyString());
        doReturn(1d).when(getMatchingModels()).getClassificationError(any(Classifier.class), any(Instances.class));

        getService().generateModel(getState());
        verify(getMatchingModels()).getAdaBoost(any(Instances.class));
        verify(getMatchingModels()).getJ48(any(Instances.class));
        verify(getMatchingModels()).getKNN(any(Instances.class));
        verify(getMatchingModels()).getNaiveBayes(any(Instances.class));
        verify(getMatchingModels()).getRandomForest(any(Instances.class));
        verify(getMatchingModels()).getLogistic(any(Instances.class));
        verify(getModelRepository()).save(any(ScoredModel.class));
        verify(getState()).isCurrentlyLearning();
        verify(getState()).setCurrentlyLearning(true);
        verify(getState()).setCurrentlyLearning(false);
    }

    @Test(expected = IllegalStateException.class)
    public void doNotGenerateModelWhenNoBrandClassifier() throws IOException {
        doReturn(false).when(getModelRepository()).brandClassifierExists();

        getService().generateModel(getState());

    }
}