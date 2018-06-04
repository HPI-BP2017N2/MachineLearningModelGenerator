package de.hpi.modelgenerator.services;


import de.hpi.machinelearning.persistence.LabeledModel;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
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
import weka.core.Instances;

@Service
@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
@Slf4j
class MatchingModels {

    private static final String NAIVE_BAYES = "naiveBayes";
    private static final String LOGISTIC = "logistic";
    private static final String RANDOM_FOREST = "randomForest";
    private static final String K_NN = "kNN";
    private static final String LINEAR_REGRESSION = "linearRegression";
    private static final String J48 = "j48";
    private static final String ADA_BOOST = "adaBoost";

    LabeledModel getNaiveBayes(Instances trainingSet) {
        Classifier cModel = new NaiveBayes();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, NAIVE_BAYES);
    }

    LabeledModel getLogistic(Instances trainingSet) {
        Classifier cModel = new Logistic();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, LOGISTIC);
    }

    LabeledModel getRandomForest(Instances trainingSet) {
        Classifier cModel = new RandomForest();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, RANDOM_FOREST);
    }

    LabeledModel getKNN(Instances trainingSet) {
        Classifier cModel = new IBk();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, K_NN);
    }

    public LabeledModel getLinearRegression(Instances trainingSet) {
        Classifier cModel = new LinearRegression();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, LINEAR_REGRESSION);
    }

    LabeledModel getJ48(Instances trainingSet) {
        Classifier cModel = new J48();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, J48);
    }

    LabeledModel getAdaBoost(Instances trainingSet) {
        Classifier cModel = new AdaBoostM1();
        try {
            cModel.buildClassifier(trainingSet);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return new LabeledModel(cModel, ADA_BOOST);
    }

    double getClassificationError(Classifier cModel, Instances trainingSet) {
        try {
            Evaluation eTest = new Evaluation(trainingSet);
            eTest.evaluateModel(cModel, trainingSet);
            return eTest.errorRate();

        } catch (Exception e) {
            e.printStackTrace();
        }

        return 1d;
    }
}
