package de.hpi.modelgenerator.services;

import de.hpi.modelgenerator.dto.ScoredModel;
import de.hpi.modelgenerator.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

@Service
@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
public class MatchingModels {

    private static final Logger log = LoggerFactory.getLogger(MatchingModels.class);

    public static LabeledModel getNaiveBayes(Instances trainingSet) {
        Classifier cModel = new NaiveBayes();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, "naiveBayes");
    }

    public static LabeledModel getLogistic(Instances trainingSet) {
        Classifier cModel = new Logistic();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, "logistic");
    }

    public static LabeledModel getRandomForest(Instances trainingSet) {
        Classifier cModel = new RandomForest();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, "randomForest");
    }

    public static LabeledModel getKNN(Instances trainingSet) {
        Classifier cModel = new IBk();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, "kNN");
    }

    public static LabeledModel getLinearRegression(Instances trainingSet) {
        Classifier cModel = new LinearRegression();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, "linearRegression");
    }

    public static LabeledModel getJ48(Instances trainingSet) {
        Classifier cModel = new J48();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, "j48");
    }

    public static LabeledModel getAdaBoost(Instances trainingSet) {
        Classifier cModel = new AdaBoostM1();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, "adaBoost");
    }

    public static double evaluateModel(Classifier cModel, Instances trainingSet) {
        Evaluation eTest;
        Instances isTestingSet = createSet(100);
        try {
            eTest = new Evaluation(trainingSet);
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

        return 1d;
    }


    public static Instances createSet(int count) {
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
