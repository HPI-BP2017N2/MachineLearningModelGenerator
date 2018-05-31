package de.hpi.modelgenerator.persistence.repo;

import de.hpi.machinelearning.persistence.ScoredModel;
import de.hpi.machinelearning.persistence.SerializedParagraphVectors;
import lombok.AccessLevel;
import lombok.Getter;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.stereotype.Repository;

import java.io.IOException;

import static org.springframework.data.mongodb.core.query.Criteria.where;
import static org.springframework.data.mongodb.core.query.Query.query;

@Repository
@Getter(AccessLevel.PRIVATE)
public class ModelRepository {

    public static final String BRAND = "brand";
    @Autowired
    @Qualifier(value = "modelTemplate")
    private MongoTemplate mongoTemplate;

    public void save(ParagraphVectors model, String type) throws IOException {
        getMongoTemplate().save(new SerializedParagraphVectors(model, type));
    }

    public void save(ScoredModel model) {
        getMongoTemplate().save(model);
    }

    public ParagraphVectors getBrandClassifier() throws IOException {
        return getMongoTemplate().findById(BRAND, SerializedParagraphVectors.class).getNeuralNetwork();
    }

    public boolean brandClassifierExists() {
        return getMongoTemplate().exists(query(where("_id").is(BRAND)), SerializedParagraphVectors.class);
    }
}
