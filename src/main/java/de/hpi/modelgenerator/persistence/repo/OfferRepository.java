package de.hpi.modelgenerator.persistence.repo;

import de.hpi.modelgenerator.persistence.ShopOffer;

import java.util.List;

public interface OfferRepository {

    List<ShopOffer> getOffers(long shopId);

}
