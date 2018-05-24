package de.hpi.modelgenerator.persistence.repo;

import de.hpi.modelgenerator.persistence.SerializedParagraphVectors;
import lombok.Getter;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.stereotype.Repository;

import static org.springframework.data.mongodb.core.query.Criteria.where;
import static org.springframework.data.mongodb.core.query.Query.query;

@Repository
@Getter
public class ModelRepositoryImpl implements ModelRepository {

    @Autowired
    @Qualifier(value = "modelTemplate")
    private MongoTemplate mongoTemplate;

    @Override
    public void save(SerializedParagraphVectors model) {
        getMongoTemplate().save(model);
    }

    @Override
    public ParagraphVectors loadModel(String modelType) {
        return getMongoTemplate().findById(modelType, SerializedParagraphVectors.class).getNeuralNetwork();
    }
}
