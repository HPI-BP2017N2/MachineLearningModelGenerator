package de.hpi.modelgenerator.services;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
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

import java.util.ArrayList;

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
            System.out.println(eTest.confusionMatrix());
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public Instances createSet(int count) {
        ArrayList<Attribute> features = AttributeVectorCreator.createFastVector();
        Instances isSet = new Instances("Rel", features, 10);
        isSet.setClassIndex(2);
        for (int i = 0; i < count; i++) {
            Instance iExample = new DenseInstance(4);
            iExample.setValue(features.get(0), 1.0);
            iExample.setValue(features.get(1), "danial");
            iExample.setValue(features.get(2), "true");
            isSet.add(iExample);
        }
        return isSet;
    }
}
