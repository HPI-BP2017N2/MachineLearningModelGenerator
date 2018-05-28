package de.hpi.modelgenerator.persistence.repo;

import de.hpi.machinelearning.persistence.SerializedParagraphVectors;
import de.hpi.machinelearning.persistence.persistence.ScoredModel;
import lombok.Getter;
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
    public void save(ScoredModel model) {
        getMongoTemplate().save(model);
    }

    @Override
    public boolean categoryClassifierExists() {
        return classifierExists("category");
    }

    @Override
    public boolean brandClassifierExists() {
        return classifierExists("brand");
    }

    @Override
    public boolean modelExists() {
        return getMongoTemplate().exists(query(where("_id").exists(true)), ScoredModel.class);
    }

    private boolean classifierExists(String classifierType) {
        return getMongoTemplate().exists(query(where("_id").is(classifierType)), SerializedParagraphVectors.class);
    }
}
