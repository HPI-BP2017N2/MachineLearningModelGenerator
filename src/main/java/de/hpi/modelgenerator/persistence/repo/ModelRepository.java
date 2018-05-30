package de.hpi.modelgenerator.persistence.repo;

import de.hpi.machinelearning.persistence.ScoredModel;
import de.hpi.machinelearning.persistence.SerializedParagraphVectors;
import lombok.Getter;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.stereotype.Repository;

import java.io.IOException;

@Repository
@Getter
public class ModelRepository {

    @Autowired
    @Qualifier(value = "modelTemplate")
    private MongoTemplate mongoTemplate;

    public void save(ParagraphVectors model, String type) throws IOException {
        getMongoTemplate().save(new SerializedParagraphVectors(model, type));
    }

    public void save(ScoredModel model) {
        getMongoTemplate().save(model);
    }

}
