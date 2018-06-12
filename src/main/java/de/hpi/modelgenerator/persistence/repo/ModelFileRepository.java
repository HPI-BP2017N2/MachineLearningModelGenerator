package de.hpi.modelgenerator.persistence.repo;

import com.fasterxml.jackson.databind.ObjectMapper;
import de.hpi.machinelearning.persistence.ScoredModel;
import de.hpi.machinelearning.persistence.SerializedParagraphVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.springframework.stereotype.Repository;

import java.io.File;
import java.io.IOException;

@Repository
public class ModelFileRepository implements ModelRepository{

    @Override
    public void save(ParagraphVectors model, String type) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        String path = System.getProperty("user.dir");
        new File(path + "/models").mkdirs();
        SerializedParagraphVectors serializedParagraphVectors = new SerializedParagraphVectors(model, type);
        mapper.writeValue(new File(path + "/models/" + type + ".json"), serializedParagraphVectors);
    }

    @Override
    public void save(ScoredModel model) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        String path = System.getProperty("user.dir");
        new File(path + "/models").mkdirs();
        mapper.writeValue(new File(path + "/models/model.json"), model);

    }

    @Override
    public ParagraphVectors getBrandClassifier() throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        String path = System.getProperty("user.dir");
        SerializedParagraphVectors classifier = mapper.readValue(new File(path + "/models/brand.json"), SerializedParagraphVectors.class);
        return classifier.getNeuralNetwork();
    }

    @Override
    public boolean brandClassifierExists() {
        String path = System.getProperty("user.dir");
        return (new File(path + "/models/brand.json")).exists();
    }
}
