package de.hpi.modelgenerator.persistence.repo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class ModelRepositoryImpl implements ModelRepository {

    @Autowired
    @Qualifier(value = "modelTemplate")
    private MongoTemplate mongoTemplate;

}
