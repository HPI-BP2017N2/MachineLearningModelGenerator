package de.hpi.modelgenerator.persistence.repo;

import de.hpi.modelgenerator.persistence.ShopOffer;
import de.hpi.modelgenerator.properties.CacheProperties;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.retry.annotation.Backoff;
import org.springframework.retry.annotation.Retryable;
import org.springframework.stereotype.Repository;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;

@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
@Repository
public class Cache {

    private final RestTemplate restTemplate;

    private final CacheProperties properties;

    @Autowired
    public Cache(RestTemplateBuilder restTemplateBuilder, CacheProperties cacheProperties) {
        this.properties = cacheProperties;
        this.restTemplate = restTemplateBuilder.build();
    }

    @Retryable(
            value = {HttpClientErrorException.class },
            maxAttempts = 5,
            backoff = @Backoff(delay = 5000))
    public ShopOffer getOffer(long shopId, String offerKey) {
        return getRestTemplate().getForObject(getOffersURI(shopId, offerKey), ShopOffer.class);
    }


    private URI getOffersURI(long shopID, String offerKey) {
        return UriComponentsBuilder.fromUriString(getProperties().getUri())
                .path(getProperties().getGetOfferRoute() + shopID)
                .queryParam("offerKey", offerKey)
                .build()
                .encode()
                .toUri();
    }



}