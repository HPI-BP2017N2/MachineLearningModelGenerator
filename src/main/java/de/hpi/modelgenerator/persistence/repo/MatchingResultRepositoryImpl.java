package de.hpi.modelgenerator.persistence.repo;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class MatchingResultRepositoryImpl implements MatchingResultRepository {

    @Autowired
    @Qualifier(value = "matchingResultTemplate")
    private MongoTemplate mongoTemplate;
}
