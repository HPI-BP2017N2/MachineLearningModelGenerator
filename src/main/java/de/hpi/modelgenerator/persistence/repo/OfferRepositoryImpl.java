package de.hpi.modelgenerator.persistence.repo;

import de.hpi.modelgenerator.persistence.ShopOffer;
import lombok.AccessLevel;
import lombok.Getter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.stereotype.Repository;

import java.util.List;


@Repository
@Getter(AccessLevel.PRIVATE)
public class OfferRepositoryImpl implements OfferRepository {

    @Autowired
    @Qualifier(value = "matchingResultTemplate")
    private MongoTemplate mongoTemplate;

    @Override
    public List<ShopOffer> getOffers(long shopId) {
        return getMongoTemplate().findAll(ShopOffer.class, Long.toString(shopId));
    }
}