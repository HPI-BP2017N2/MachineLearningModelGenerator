package de.hpi.modelgenerator.services;

import de.hpi.modelgenerator.persistence.AttributeVector;
import de.hpi.modelgenerator.persistence.FeatureInstance;
import de.hpi.modelgenerator.persistence.ParsedOffer;
import de.hpi.modelgenerator.persistence.ShopOffer;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.bytedeco.javacpp.presets.opencv_core;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

@Service
@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
public class WEKA {

    private static final Logger log = LoggerFactory.getLogger(WEKA.class);
    private Instances TrainingSet;

    public Classifier calculateModel() {
        Instances isTrainingSet = createSet(10);
        setTrainingSet(isTrainingSet);
        Classifier cModel = new NaiveBayes();
        try {
            cModel.buildClassifier(isTrainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return cModel;
    }

    public void evaluateModel() {
        Classifier cModel = calculateModel();
        Evaluation eTest;
        Instances isTestingSet = createSet(100);
        try {
            eTest = new Evaluation(getTrainingSet());
            eTest.evaluateModel(cModel, isTestingSet);
            String strSummary = eTest.toSummaryString();
            System.out.println(strSummary);

            System.out.println("\nConfusion Matrix: ");

            for(double[] array : eTest.confusionMatrix()) {
                for(double i : array) {
                    System.out.printf(String.format("%1$" + 4 + "s", i) + "\t");
                }
                System.out.printf("\n");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Instances createSet(int count) {
        ArrayList<Attribute> features = new AttributeVector();
        Instances isSet = new Instances("Rel", features, count);

        ShopOffer shopOffer = new ShopOffer();
        ParsedOffer parsedOffer = new ParsedOffer();
        Map<String, String> title = new HashMap<>();
        title.put("0", "iPhone7");
        shopOffer.setTitles(title);
        parsedOffer.setTitle("iPhone7");
        shopOffer.setDescriptions(title);
        parsedOffer.setTitle("iPhone7");
        parsedOffer.setBrandName("Apple");
        shopOffer.setBrandName("Apple");
        parsedOffer.setPrice("1000");
        Map<String, Double> price = new HashMap<>();
        price.put("0", 1000d);
        shopOffer.setPrices(price);
        parsedOffer.setPrice("1000");
        shopOffer.setMappedCatalogCategory("12345");
        parsedOffer.setCategory("12345");
        Map<String, String> url = new HashMap<>();
        url.put("0", "http://example.com/123");
        shopOffer.setUrls(url);
        parsedOffer.setUrl("http://example.com/123");
        shopOffer.setImageId("qwertz");
        parsedOffer.setImageUrl( "qwerty");

        for (int i = 0; i < count; i++) {
            Instance iExample = new FeatureInstance(shopOffer, parsedOffer, true);
            isSet.add(iExample);
        }

        isSet.setClassIndex(features.size() - 1);
        return isSet;
    }
}
