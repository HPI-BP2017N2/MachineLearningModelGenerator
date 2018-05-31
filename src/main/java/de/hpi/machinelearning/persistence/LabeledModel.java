package de.hpi.machinelearning.persistence;


import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;

@Getter
@Setter
@RequiredArgsConstructor
public class LabeledModel {

    private final Classifier model;
    private final String modelType;

    public ScoredModel toScoredModel(double score) {
        OutputStream out = new ByteArrayOutputStream(100000000);

        try {
            SerializationHelper.write(out, getModel());
        } catch (Exception e) {
            e.printStackTrace();
        }

         byte[] model = ((ByteArrayOutputStream) out).toByteArray();
        return new ScoredModel(model, getModelType(), score);
    }
}
